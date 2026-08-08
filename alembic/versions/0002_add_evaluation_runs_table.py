from alembic import op
import sqlalchemy as sql
from sqlalchemy.dialects.postgresql import UUID, JSON

revision = "0002"
down_revision = "0001"

def upgrade():
    op.create_table(
        "evaluation_runs",
        sql.Column("id", UUID(as_uuid=True), primary_key=True),
        sql.Column("dataset_id", UUID(as_uuid=True), nullable=False),                                                                                  
        sql.Column("model_id", sql.String, nullable=False),
        sql.Column("provider", sql.String, nullable=False),                                                                                             
        sql.Column("status", sql.String, nullable=False, server_default="pending"),                                                                   
        sql.Column("num_samples", sql.Integer, nullable=True),
        sql.Column("num_errors", sql.Integer, nullable=True),
        sql.Column("error_rate", sql.Float, nullable=True),                                                                                             
        sql.Column("avg_latency_ms", sql.Float, nullable=True),                                                                                       
        sql.Column("total_cost_usd", sql.Float, nullable=True),
        sql.Column("duration_seconds", sql.Float, nullable=True),                                                                                       
        sql.Column("aggregate_metrics", JSON, nullable=True),
        sql.Column("sample_results", JSON, nullable=True),                                                                                             
        sql.Column("config", JSON, nullable=True),                                                                                                   
        sql.Column("error_message", sql.String, nullable=True),
        sql.Column("created_at", sql.DateTime(timezone=True), server_default=sql.func.now(), nullable=False),
        sql.Column("updated_at", sql.DateTime(timezone=True), server_default=sql.func.now(), nullable=False),
        sql.Column("completed_at", sql.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_eval_runs_model_id", "evaluation_runs", ["model_id"])                                                                         
    op.create_index("ix_eval_runs_status", "evaluation_runs", ["status"])

def downgrade():
    op.drop_table("evaluation_runs")