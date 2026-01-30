from database import engine, Base
from sqlalchemy import text

print("⚠️  REBUILDING DATABASE SCHEMA...")

with engine.connect() as con:
    try:
        # We drop the tables so they can be recreated with the new 'is_processed' column
        con.execute(text("DROP TABLE IF EXISTS comparison_candidates CASCADE"))
        con.execute(text("DROP TABLE IF EXISTS comparison_sessions CASCADE"))
        con.execute(text("DROP TABLE IF EXISTS resumes CASCADE"))
        con.commit()
        print("✅ Old tables dropped.")
    except Exception as e:
        print(f"Error: {e}")

# Create new tables with the correct columns
Base.metadata.create_all(bind=engine)
print("🚀 SUCCESS! Database is fixed.") 