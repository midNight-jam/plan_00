from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Float, Integer, DateTime
import datetime
import uuid
import asyncio

# Connects to the 'forge-db' container from our docker-compose
DATABASE_URL = "postgresql+asyncpg://forge:forgepassword@db:5432/forgedb"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    model_name = Column(String, nullable=False)
    prompt_length = Column(Integer, nullable=False)  # Rough char count
    generation_time_ms = Column(Float, nullable=False)

async def init_db():
    """
    Creates the tables in Postgres if they don't exist.
    Includes a retry loop to wait for the database container to fully boot.
    """
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            async with engine.begin() as conn:
                # This explicitly generates the SQL to create your tables
                await conn.run_sync(Base.metadata.create_all)
            print("✅ Database tables verified and created successfully!")
            break  # If successful, break out of the retry loop
            
        except Exception as e:
            if attempt == max_retries - 1:
                print("❌ Failed to connect to PostgreSQL after multiple attempts.")
                raise e
            print(f"⏳ Waiting for PostgreSQL to be ready... (Attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(3) # Wait 3 seconds before trying again