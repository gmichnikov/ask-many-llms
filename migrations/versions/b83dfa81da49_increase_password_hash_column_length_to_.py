"""increase password_hash column length to 255

Revision ID: b83dfa81da49
Revises: 96006dcc6083
Create Date: 2025-09-24 09:15:26.089584

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b83dfa81da49'
down_revision = '96006dcc6083'
branch_labels = None
depends_on = None


def upgrade():
    # Increase password_hash column length from 128 to 255 characters
    op.alter_column('user', 'password_hash',
                    existing_type=sa.VARCHAR(length=128),
                    type_=sa.String(length=255),
                    existing_nullable=True)


def downgrade():
    # Revert password_hash column length back to 128 characters
    op.alter_column('user', 'password_hash',
                    existing_type=sa.VARCHAR(length=255),
                    type_=sa.String(length=128),
                    existing_nullable=True)
