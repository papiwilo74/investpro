"""Script de verificación de conexión a Neon PostgreSQL."""

import os
import sys


def check_neon_connection():
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        print("❌ Error: La variable de entorno 'DATABASE_URL' no está definida.")
        print("Asegúrate de configurar DATABASE_URL con la cadena de conexión de Neon DB.")
        sys.exit(1)

    print("🔍 Intentando conectar con Neon PostgreSQL...")
    try:
        import psycopg2

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print("✅ ¡Conexión exitosa a Neon DB!")
        print(f"📌 Información del servidor: {db_version[0]}")
        cur.close()
        conn.close()
    except ImportError:
        # Si no está instalado psycopg2, intentamos con sqlalchemy o asyncpg
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(db_url)
            with engine.connect() as connection:
                result = connection.execute(text("SELECT version();"))
                row = result.fetchone()
                print("✅ ¡Conexión exitosa a Neon DB (vía SQLAlchemy)!")
                print(f"📌 Información del servidor: {row[0]}")
        except Exception as err:
            print(f"❌ Error al conectar a Neon DB: {err}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error de conexión a Neon DB: {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_neon_connection()
