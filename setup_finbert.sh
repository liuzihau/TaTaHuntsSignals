#!/bin/bash

##########################################################
# FinBERT Setup Script for Phase 4-5 Redux
# Installs dependencies and initializes database
##########################################################

set -e  # Exit on error

echo "=========================================="
echo "FinBERT Setup Script"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    OS="unknown"
fi

echo -e "${YELLOW}Detected OS: $OS${NC}"

##########################################################
# Step 1: Install Python Dependencies
##########################################################
echo ""
echo -e "${YELLOW}Step 1: Installing Python dependencies...${NC}"

if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    pip3 install --user -r requirements.txt || {
        echo -e "${RED}Failed to install from requirements.txt${NC}"
        echo "Attempting to install individual packages..."
        pip3 install --user transformers torch tqdm psycopg2-binary
    }
else
    echo "Installing individual packages..."
    pip3 install --user transformers torch tqdm psycopg2-binary
fi

if python3 -c "import transformers" 2>/dev/null; then
    echo -e "${GREEN}✓ transformers installed${NC}"
else
    echo -e "${RED}✗ transformers not installed${NC}"
    exit 1
fi

if python3 -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}✓ torch installed${NC}"
else
    echo -e "${RED}✗ torch not installed${NC}"
    exit 1
fi

if python3 -c "import psycopg2" 2>/dev/null; then
    echo -e "${GREEN}✓ psycopg2 installed${NC}"
else
    echo -e "${RED}✗ psycopg2 not installed${NC}"
    exit 1
fi

##########################################################
# Step 2: Verify PostgreSQL Connection
##########################################################
echo ""
echo -e "${YELLOW}Step 2: Verifying PostgreSQL connection...${NC}"

python3 << 'EOF'
import sys
try:
    from storage.postgres_client import PostgreSQLClient
    pg = PostgreSQLClient()
    pg.close()
    print("\033[0;32m✓ PostgreSQL connection successful\033[0m")
except Exception as e:
    print(f"\033[0;31m✗ PostgreSQL connection failed: {e}\033[0m")
    sys.exit(1)
EOF

##########################################################
# Step 3: Initialize Database Schema
##########################################################
echo ""
echo -e "${YELLOW}Step 3: Initializing database schema...${NC}"

DB_USER="${DB_USER:-tatahunts_user}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-tatahunts_rag}"

echo "Applying PostgreSQL schema..."
if psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -f storage/postgres_schema.sql > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Schema initialized${NC}"
elif [ -f "storage/migration_add_finbert_embedding.sql" ]; then
    echo "Applying migration for existing database..."
    psql -U "$DB_USER" -h "$DB_HOST" -p "$DB_PORT" -d "$DB_NAME" -f storage/migration_add_finbert_embedding.sql > /dev/null 2>&1
    echo -e "${GREEN}✓ Migration applied${NC}"
else
    echo -e "${RED}✗ Failed to apply schema${NC}"
    exit 1
fi

##########################################################
# Step 4: Verify Schema
##########################################################
echo ""
echo -e "${YELLOW}Step 4: Verifying schema...${NC}"

python3 << 'EOF'
import sys
try:
    from storage.postgres_client import PostgreSQLClient
    pg = PostgreSQLClient()
    
    # Check if articles table exists
    with pg._require_conn().cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'articles' AND column_name = 'embedding'
        """)
        result = cur.fetchone()
    
    if result:
        print(f"\033[0;32m✓ Embedding column exists: {result[1]}\033[0m")
    else:
        print("\033[0;31m✗ Embedding column not found\033[0m")
        sys.exit(1)
    
    pg.close()
except Exception as e:
    print(f"\033[0;31m✗ Schema verification failed: {e}\033[0m")
    sys.exit(1)
EOF

##########################################################
# Step 5: Test FinBERT Encoder
##########################################################
echo ""
echo -e "${YELLOW}Step 5: Testing FinBERT encoder...${NC}"

python3 << 'EOF'
import sys
try:
    from embedding.finbert_encoder import FinBertEncoder
    
    print("Loading FinBERT model (this may take 1-2 minutes on first run)...")
    encoder = FinBertEncoder(device="cpu")  # Use CPU for testing
    
    print("Testing single text encoding...")
    embedding = encoder.encode("Apple stock surges on earnings beat")
    
    if len(embedding) == 768:
        print(f"\033[0;32m✓ FinBERT encoder working: {len(embedding)}-dim vector\033[0m")
    else:
        print(f"\033[0;31m✗ Unexpected embedding dimension: {len(embedding)}\033[0m")
        sys.exit(1)
except Exception as e:
    print(f"\033[0;31m✗ FinBERT test failed: {e}\033[0m")
    sys.exit(1)
EOF

##########################################################
# Step 6: Summary
##########################################################
echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "✓ Python dependencies installed"
echo "✓ PostgreSQL connected"
echo "✓ Database schema initialized"
echo "✓ Embedding column added"
echo "✓ FinBERT encoder tested"
echo ""
echo "Next steps:"
echo "1. Encode articles: python3 -c \"from jobs.embedding_job import EmbeddingJob; from storage.postgres_client import PostgreSQLClient; from embedding.finbert_encoder import FinBertEncoder; job = EmbeddingJob(PostgreSQLClient(), FinBertEncoder()); print(job.embed_new_articles())\""
echo "2. Monitor encoding: tail -f embedding.log"
echo "3. Verify embeddings: psql -U $DB_USER -d $DB_NAME -c \"SELECT COUNT(*) FROM articles WHERE embedding IS NOT NULL;\""
echo ""
echo "For more info, see: FINBERT_SETUP.md"
echo ""
