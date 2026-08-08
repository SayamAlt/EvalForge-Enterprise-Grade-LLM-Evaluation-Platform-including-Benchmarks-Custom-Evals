"""
Sprint 3 — Provider smoke test.

Calls each configured provider with a minimal prompt and reports
latency, token usage, and cost. Skips providers with no API key.
Ollama is started automatically if not running.

Usage:
    uv run python scripts/smoke_test_providers.py
    uv run python scripts/smoke_test_providers.py --providers openai anthropic groq
"""
import asyncio, argparse, os, subprocess, time, sys, httpx
from dotenv import load_dotenv
from evalforge.providers.registry import get_provider
from evalforge.providers.base import GenerationRequest

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

PROVIDER_MODELS = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "google":    "gemini-2.0-flash",
    "groq":      "llama-3.1-8b-instant",
    "deepseek":  "deepseek-chat",
    "qwen":      "qwen-turbo",
    "moonshot":  "moonshot-v1-8k",
    "minimax":   "abab6.5s",
    "mistral":   "mistral-small",
    "writer":    "palmyra-x-003",
    "ollama":    "llama3.2"
}

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

PROMPT = GenerationRequest(
    prompt="Reply with exactly three words: 'EvalForge works perfectly'",
    max_tokens=20,
    temperature=0.0,
)

ENV_KEYS = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google":    "GOOGLE_API_KEY",
    "groq":      "GROQ_API_KEY",
    "deepseek":  "DEEPSEEK_API_KEY",
    "qwen":      "DASHSCOPE_API_KEY",
    "moonshot":  "MOONSHOT_API_KEY",
    "minimax":   "MINIMAX_API_KEY",
    "mistral":   "MISTRAL_API_KEY",
    "writer":    "WRITER_API_KEY",
    "ollama":    None,
}

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def _ollama_running() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False

def _ollama_has_model(model: str) -> bool:
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        tags = r.json().get("models", [])
        return any(m.get("name", "").split(":")[0] == model.split(":")[0] for m in tags)
    except Exception:
        return False

def ensure_ollama(model: str) -> str | None:
    """Start Ollama and pull model if needed. Returns error string or None."""
    if not _ollama_running():
        print(f"{BLUE}⟳ ollama       starting ollama serve ...{RESET}")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return "ollama binary not found — install from https://ollama.com"

        # Wait up to 10s for daemon to be ready
        for _ in range(20):
            time.sleep(0.5)
            if _ollama_running():
                break
        else:
            return "ollama serve started but not responding after 10s"

    if not _ollama_has_model(model):
        print(f"{BLUE}⟳ ollama       pulling {model} (first run, may take a while) ...{RESET}")
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return f"ollama pull {model} failed: {result.stderr.strip()[:100]}"

    return None

async def test_provider(name: str, model: str) -> dict:
    env_key = ENV_KEYS.get(name)
    if env_key and not os.getenv(env_key):
        return {"name": name, "model": model, "status": "skipped", "reason": f"{env_key} not set"}

    if name == "ollama":
        err = ensure_ollama(model)
        if err:
            return {"name": name, "model": model, "status": "error", "reason": err}

    try:
        provider = get_provider(name, model)
        response = await provider.generate(PROMPT)
        cost = provider.estimate_cost(response)
        return {
            "name": name,
            "model": model,
            "status": "ok",
            "text": response.text.strip(),
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": cost,
        }
    except Exception as e:
        return {"name": name, "model": model, "status": "error", "reason": str(e)[:120]}

async def run(providers: list[str]) -> None:
    print(f"\n{BOLD}EvalForge — Provider Smoke Test{RESET}")
    print("=" * 62)

    # Run Ollama setup synchronously before async tasks (subprocess blocks)
    tasks = [test_provider(name, PROVIDER_MODELS[name]) for name in providers]
    results = await asyncio.gather(*tasks)

    passed = failed = skipped = 0
    for r in results:
        if r["status"] == "ok":
            passed += 1
            cost_str = f"${r['cost_usd']:.6f}" if r["cost_usd"] else "free"
            print(
                f"{GREEN}✓ {r['name']:<12}{RESET}"
                f"  {r['model']:<32}"
                f"  {r['latency_ms']:>6.0f}ms"
                f"  {r['input_tokens']:>4}→{r['output_tokens']:<4} tok"
                f"  {cost_str}"
            )
            print(f"   └─ response: {r['text']!r}")
        elif r["status"] == "skipped":
            skipped += 1
            print(f"{YELLOW}– {r['name']:<12}{RESET}  {r['model']:<32}  skipped ({r['reason']})")
        else:
            failed += 1
            print(f"{RED}✗ {r['name']:<12}{RESET}  {r['model']:<32}  ERROR: {r['reason']}")

    print("=" * 62)
    print(f"{BOLD}Results: {GREEN}{passed} passed{RESET}  {RED}{failed} failed{RESET}  {YELLOW}{skipped} skipped{RESET}\n")

def main():
    parser = argparse.ArgumentParser(description="Smoke-test EvalForge LLM providers")
    parser.add_argument(
        "--providers", nargs="+", choices=list(PROVIDER_MODELS), default=list(PROVIDER_MODELS),
        help="Providers to test (default: all)"
    )
    args = parser.parse_args()
    asyncio.run(run(args.providers))

if __name__ == "__main__":
    main()