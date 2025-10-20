import json
import sqlite3
import os
from datetime import datetime

def create_database_schema(cursor):
    """Cria a tabela para armazenar os dados das empresas"""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS empresas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            empresa TEXT NOT NULL,
            setor TEXT NOT NULL,
            faturamento_2024 REAL,
            lucro_liquido_2024 REAL,
            contas_a_receber REAL,
            pdd_balanco_2024 REAL,
            pdd_fc_2024 REAL,
            data_importacao TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Criar índices para melhorar performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_empresa ON empresas(empresa)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_setor ON empresas(setor)')

def import_json_to_sqlite(json_file_path, db_path='empresas.db'):
    """Importa dados do JSON para SQLite"""
    
    # Conectar ao banco SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Criar schema da tabela
        create_database_schema(cursor)
        
        # Ler dados do JSON
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        print(f"Encontrados {len(data)} registros no arquivo JSON")
        
        # Limpar tabela existente (opcional - remova se quiser manter dados antigos)
        cursor.execute('DELETE FROM empresas')
        
        # Inserir dados
        inserted_count = 0
        for empresa in data:
            cursor.execute('''
                INSERT INTO empresas (
                    empresa, setor, faturamento_2024, lucro_liquido_2024, 
                    contas_a_receber, pdd_balanco_2024, pdd_fc_2024
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                empresa.get('Empresa'),
                empresa.get('Setor'),
                empresa.get('Faturamento 2024 (R$ mil)'),
                empresa.get('Lucro Líquido 2024 (R$ mil)'),
                empresa.get('Contas a Receber (R$ mil)'),
                empresa.get('PDD - Balanço 2024 (R$ mil)'),
                empresa.get('PDD - FC 2024 (R$ mil)')
            ))
            inserted_count += 1
        
        # Commit das alterações
        conn.commit()
        print(f"✅ {inserted_count} registros inseridos com sucesso!")
        
        # Mostrar estatísticas
        cursor.execute('SELECT COUNT(*) FROM empresas')
        total_records = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT setor) FROM empresas')
        total_sectors = cursor.fetchone()[0]
        
        print(f"📊 Estatísticas:")
        print(f"   - Total de empresas: {total_records}")
        print(f"   - Total de setores: {total_sectors}")
        
        # Mostrar setores únicos
        cursor.execute('SELECT DISTINCT setor FROM empresas ORDER BY setor')
        sectors = cursor.fetchall()
        print(f"\n📋 Setores encontrados:")
        for sector in sectors:
            print(f"   - {sector[0]}")
        
    except Exception as e:
        print(f"❌ Erro durante a importação: {e}")
        conn.rollback()
    finally:
        conn.close()

def query_database(db_path='empresas.db'):
    """Função para consultar o banco de dados"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Exemplo de consultas
        print("\n🔍 Consultas de exemplo:")
        
        # Total por setor
        cursor.execute('''
            SELECT setor, COUNT(*) as total_empresas 
            FROM empresas 
            GROUP BY setor 
            ORDER BY total_empresas DESC
        ''')
        
        print("\n📈 Empresas por setor:")
        for row in cursor.fetchall():
            print(f"   {row[0]}: {row[1]} empresas")
        
        # Empresas com maior PDD
        cursor.execute('''
            SELECT empresa, setor, pdd_balanco_2024 
            FROM empresas 
            WHERE pdd_balanco_2024 IS NOT NULL 
            ORDER BY pdd_balanco_2024 DESC 
            LIMIT 10
        ''')
        
        print("\n💰 Top 10 empresas com maior PDD (Balanço):")
        for row in cursor.fetchall():
            print(f"   {row[0]} ({row[1]}): R$ {row[2]:,.0f} mil")
            
    except Exception as e:
        print(f"❌ Erro durante consulta: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Caminho para o arquivo JSON
    json_file = "Resultados/setores_atualizados.json"
    
    if not os.path.exists(json_file):
        print(f"❌ Arquivo não encontrado: {json_file}")
        exit(1)
    
    print("🚀 Iniciando importação para SQLite...")
    print(f"📁 Arquivo fonte: {json_file}")
    print(f"💾 Banco de dados: empresas.db")
    
    # Importar dados
    import_json_to_sqlite(json_file)
    
    # Executar consultas de exemplo
    query_database()
    
    print(f"\n✅ Importação concluída! Banco de dados criado: empresas.db")
    print(f"💡 Você pode usar ferramentas como DB Browser for SQLite para visualizar os dados")
