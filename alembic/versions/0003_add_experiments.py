from alembic import op
import sqlalchemy as sql
from sqlalchemy.dialects.postgresql import UUID, JSON

revision = "0003"
down_revision = "0002"

def upgrade():
    op.create_table(                                                                                                                                
        "experiments",
        sql.Column("id", UUID(as_uuid=True), primary_key=True),
        sql.Column("name", sql.String, nullable=False),
        sql.Column("description", sql.String, nullable=True),                                                                                         
        sql.Column("dataset_id", UUID(as_uuid=True), nullable=True),
        sql.Column("benchmark_name", sql.String, nullable=True),                                                                                      
        sql.Column("primary_metric", sql.String, nullable=False, server_default="answer_match"),                                                    
        sql.Column("tags", JSON, nullable=True),                                                                                                      
        sql.Column("created_at", sql.DateTime(timezone=True), server_default=sql.func.now(), nullable=False),
        sql.Column("updated_at", sql.DateTime(timezone=True), server_default=sql.func.now(), nullable=False),                                         
    )                                                                                                                                                 
    op.create_table(
        "experiment_runs",
        sql.Column("id", UUID(as_uuid=True), primary_key=True),
        sql.Column("experiment_id", UUID(as_uuid=True), nullable=False),
        sql.Column("run_id", UUID(as_uuid=True), nullable=False),
        sql.Column("created_at", sql.DateTime(timezone=True), server_default=sql.func.now(), nullable=False),
        sql.Column("updated_at", sql.DateTime(timezone=True), server_default=sql.func.now(), nullable=False),
    )                                                                                                                                                 
    op.create_index("ix_experiment_runs_exp_id", "experiment_runs", ["experiment_id"])
    # Make dataset_id nullable + add benchmark_name to eval runs                                                                                      
    op.alter_column("evaluation_runs", "dataset_id", nullable=True)                                                                                 
    op.add_column("evaluation_runs", sql.Column("benchmark_name", sql.String, nullable=True)) 
    
def downgrade():
    op.drop_column("evaluation_runs", "benchmark_name")                                                                                             
    op.alter_column("evaluation_runs", "dataset_id", nullable=False)                                                                                
    op.drop_table("experiment_runs")                                                                                                                  
    op.drop_table("experiments")