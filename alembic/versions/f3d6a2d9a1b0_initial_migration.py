"""Initial migration with auth tables

Revision ID: f3d6a2d9a1b0
Revises: 
Create Date: 2025-09-29 18:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'f3d6a2d9a1b0'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Auth tables first
    op.create_table(
        'users',
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('password_hash', sa.String(), nullable=False),
        sa.Column('full_name', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('department', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('user_id'),
        sa.UniqueConstraint('email')
    )
    
    op.create_table(
        'refresh_tokens',
        sa.Column('token', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('token')
    )
    
    # Create indexes for performance
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_refresh_tokens_user_id', 'refresh_tokens', ['user_id'])
    op.create_index('ix_refresh_tokens_expires_at', 'refresh_tokens', ['expires_at'])
    
    # Device and patient tables
    op.create_table(
        'devices',
        sa.Column('device_id', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('device_id')
    )
    
    op.create_table(
        'patients',
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('medical_record_number', sa.String(), nullable=False),
        sa.Column('full_name', sa.String(), nullable=False),
        sa.Column('date_of_birth', sa.Date(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('patient_id'),
        sa.UniqueConstraint('medical_record_number')
    )
    
    # Sessions table
    op.create_table(
        'sessions',
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('device_id', sa.String(), nullable=True),
        sa.Column('patient_id', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True, server_default='active'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.ForeignKeyConstraint(['device_id'], ['devices.device_id'], ),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ),
        sa.PrimaryKeyConstraint('session_id')
    )
    
    # CTG samples with TimescaleDB hypertable
    op.create_table(
        'ctg_samples',
        sa.Column('session_id', sa.String(), nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('bpm', sa.Float(), nullable=True),
        sa.Column('ua', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('session_id', 'ts')
    )
    
    #  Convert to TimescaleDB hypertable
    # op.execute("""
    #     SELECT create_hypertable('ctg_samples', 'ts', 
    #         chunk_time_interval => INTERVAL '1 day',
    #         if_not_exists => TRUE
    #     );
    # """)
    
    # Risk records
    op.create_table(
        'risk_records',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(), nullable=True),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('hypoxia_prob', sa.Float(), nullable=False),
        sa.Column('band', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.session_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Events table
    op.create_table(
        'events',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('session_id', sa.String(), nullable=True),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('start_ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_ts', sa.DateTime(timezone=True), nullable=True),
        sa.Column('dur_s', sa.Float(), nullable=True),
        sa.Column('min_bpm', sa.Float(), nullable=True),
        sa.Column('max_drop', sa.Float(), nullable=True),
        sa.Column('severity', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.session_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id', 'type', 'start_ts', 'end_ts', name='uq_event_unique')
    )
    
    # Create indexes for better query performance
    op.create_index('ix_sessions_patient_id', 'sessions', ['patient_id'])
    op.create_index('ix_sessions_device_id', 'sessions', ['device_id'])
    op.create_index('ix_risk_records_session_id', 'risk_records', ['session_id'])
    op.create_index('ix_risk_records_ts', 'risk_records', ['ts'])
    op.create_index('ix_events_session_id', 'events', ['session_id'])
    op.create_index('ix_events_start_ts', 'events', ['start_ts'])

def downgrade() -> None:
    op.drop_index('ix_events_start_ts', table_name='events')
    op.drop_index('ix_events_session_id', table_name='events')
    op.drop_index('ix_risk_records_ts', table_name='risk_records')
    op.drop_index('ix_risk_records_session_id', table_name='risk_records')
    op.drop_index('ix_sessions_device_id', table_name='sessions')
    op.drop_index('ix_sessions_patient_id', table_name='sessions')
    
    op.drop_table('events')
    op.drop_table('risk_records')
    op.drop_table('ctg_samples')
    op.drop_table('sessions')
    op.drop_table('patients')
    op.drop_table('devices')
    
    op.drop_index('ix_refresh_tokens_expires_at', table_name='refresh_tokens')
    op.drop_index('ix_refresh_tokens_user_id', table_name='refresh_tokens')
    op.drop_table('refresh_tokens')
    
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')