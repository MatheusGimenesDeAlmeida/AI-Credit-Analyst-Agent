import os
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import re

# ========= CONFIGURAÇÕES =========
load_dotenv()

PDF_DIR = Path("./demonstrativos_privadas")
INDEX_NAME = "financial-reports-private"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
PINECONE_REGION = "us-east-1"
BATCH_SIZE = 100  # Chunks por lote
PDFS_PER_SESSION = None  # None = processa todos de uma vez, ou coloque 20 para processar 20 por vez

# Arquivo de controle (rastreia o que já foi processado)
TRACKING_FILE = Path("./processing_log.json")

# ========== CHAVES =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys não definidas")

# ========== SISTEMA DE TRACKING ==========
class ProcessingTracker:
    """Gerencia o histórico de processamento para evitar retrabalho"""
    
    def __init__(self, tracking_file: Path):
        self.tracking_file = tracking_file
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Carrega histórico de processamento"""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {
            "processed_files": {},
            "failed_files": {},
            "last_run": None,
            "total_chunks": 0
        }
    
    def _save(self):
        """Salva histórico"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def is_processed(self, file_path: Path, file_hash: str) -> bool:
        """Verifica se arquivo já foi processado com mesmo conteúdo"""
        filename = file_path.name
        if filename in self.data["processed_files"]:
            return self.data["processed_files"][filename]["hash"] == file_hash
        return False
    
    def mark_processed(self, file_path: Path, file_hash: str, num_chunks: int):
        """Marca arquivo como processado"""
        self.data["processed_files"][file_path.name] = {
            "hash": file_hash,
            "processed_at": datetime.now().isoformat(),
            "chunks": num_chunks
        }
        self.data["total_chunks"] += num_chunks
        self.data["last_run"] = datetime.now().isoformat()
        self._save()
    
    def mark_failed(self, file_path: Path, error: str):
        """Registra falha no processamento"""
        self.data["failed_files"][file_path.name] = {
            "error": str(error),
            "failed_at": datetime.now().isoformat()
        }
        self._save()
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas de processamento"""
        return {
            "processed": len(self.data["processed_files"]),
            "failed": len(self.data["failed_files"]),
            "total_chunks": self.data["total_chunks"]
        }

# ========== FUNÇÕES AUXILIARES ==========
def get_document_hash(pdf_path: Path) -> str:
    """Gera hash do PDF para detectar mudanças"""
    with open(pdf_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def preprocess_text(text: str) -> str:
    """Limpa e normaliza texto"""
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def identify_financial_sections(text: str) -> bool: 
    """Identifica seções financeiras relevantes"""
    keywords = [
        'balanço patrimonial', 'ativo circulante', 'contas a receber',
        'notas explicativas', 'instrumentos financeiros', 'provisões',
        'clientes', 'duplicatas a receber', 'pcld', 'pdd', 'descrição da conta',
        'dfs consolidadas', 'receita', 'demonstração do fluxo de caixa', 
        'despesas', 'demonstração do resultado', 'dre'
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)

def estimate_cost(num_chunks: int, avg_tokens_per_chunk: int = 300) -> float:
    """Estima custo de embedding (OpenAI ada-002: $0.0001/1k tokens)"""
    total_tokens = num_chunks * avg_tokens_per_chunk
    cost = (total_tokens / 1000) * 0.0001
    return cost

# ========== INICIALIZAÇÃO ==========
print("🚀 Sistema de Ingestão em Larga Escala")
print("=" * 60)

pc = Pinecone(api_key=PINECONE_API_KEY)

# Cria índice se não existir
existing = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing:
    print(f"Criando índice '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": PINECONE_REGION}}
    )
    time.sleep(10)  # Aguarda criação
    print("✅ Índice criado!")

index = pc.Index(INDEX_NAME)
tracker = ProcessingTracker(TRACKING_FILE)

# ========== DESCOBERTA DE ARQUIVOS ==========
if not PDF_DIR.exists():
    raise FileNotFoundError(f"❌ Pasta {PDF_DIR} não existe")

all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
print(f"\n📁 Total de PDFs na pasta: {len(all_pdfs)}")

# Filtra PDFs já processados
pdfs_to_process = []
for pdf in all_pdfs:
    file_hash = get_document_hash(pdf)
    if not tracker.is_processed(pdf, file_hash):
        pdfs_to_process.append(pdf)

stats = tracker.get_stats()
print(f"✅ Já processados: {stats['processed']}")
print(f"❌ Com falhas: {stats['failed']}")
print(f"📊 Total de chunks indexados: {stats['total_chunks']}")
print(f"🆕 Arquivos novos para processar: {len(pdfs_to_process)}")

if not pdfs_to_process:
    print("\n✨ Todos os arquivos já foram processados!")
    exit(0)

# Limita quantidade por sessão para segurança
if PDFS_PER_SESSION:
    pdfs_this_session = pdfs_to_process[:PDFS_PER_SESSION]
    print(f"\n⚡ Processando {len(pdfs_this_session)} PDFs nesta sessão")
    print(f"   (Restantes: {len(pdfs_to_process) - len(pdfs_this_session)})")
else:
    pdfs_this_session = pdfs_to_process
    print(f"\n⚡ Processando TODOS os {len(pdfs_this_session)} PDFs nesta sessão")

# ========== PROCESSAMENTO ==========
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator=True
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

total_chunks_session = 0
start_time = time.time()

for idx, pdf_path in enumerate(pdfs_this_session, 1):
    print(f"\n{'='*60}")
    print(f"📄 [{idx}/{len(pdfs_this_session)}] {pdf_path.name}")
    print(f"{'='*60}")
    
    try:
        company = pdf_path.stem
        file_hash = get_document_hash(pdf_path)
        
        # Carrega e processa PDF
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        texts = []
        metadatas = []
        
        for page_idx, page in enumerate(pages, start=1):
            page_text = preprocess_text(page.page_content or "")
            if not page_text:
                continue
            
            is_relevant = identify_financial_sections(page_text)
            chunks = splitter.split_text(page_text)
            
            for i, chunk in enumerate(chunks, start=1):
                texts.append(chunk)
                metadatas.append({
                    "company": company,
                    "source_file": pdf_path.name,
                    "page": page_idx,
                    "chunk_index": i,
                    "is_financial_section": is_relevant,
                    "document_hash": file_hash,
                    "processed_at": datetime.now().isoformat()
                })
        
        if not texts:
            print("⚠️  Nenhum conteúdo extraído")
            tracker.mark_failed(pdf_path, "No content extracted")
            continue
        
        print(f"📊 Gerados {len(texts)} chunks")
        
        # Estima custo
        cost = estimate_cost(len(texts))
        print(f"💰 Custo estimado: ${cost:.4f}")
        
        # Indexa em lotes
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_metadatas = metadatas[i:i+BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            
            print(f"  📦 Lote {batch_num}/{total_batches}: {len(batch_texts)} chunks...", end=" ")
            
            try:
                vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
                print("✅")
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"❌ Erro: {e}")
                tracker.mark_failed(pdf_path, f"Batch {batch_num} failed: {e}")
                raise
        
        # Marca como processado
        tracker.mark_processed(pdf_path, file_hash, len(texts))
        total_chunks_session += len(texts)
        print(f"✅ {pdf_path.name} indexado com sucesso!")
        
    except Exception as e:
        print(f"❌ ERRO ao processar {pdf_path.name}: {e}")
        tracker.mark_failed(pdf_path, str(e))
        continue

# ========== RESUMO FINAL ==========
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print("📊 RESUMO DA SESSÃO")
print(f"{'='*60}")
print(f"⏱️  Tempo total: {elapsed/60:.1f} minutos")
print(f"📄 PDFs processados: {len(pdfs_this_session)}")
print(f"📊 Chunks indexados: {total_chunks_session}")
print(f"💰 Custo estimado: ${estimate_cost(total_chunks_session):.4f}")

final_stats = tracker.get_stats()
print(f"\n🎯 TOTAL GERAL:")
print(f"   ✅ Processados: {final_stats['processed']}")
print(f"   📊 Total chunks: {final_stats['total_chunks']}")
print(f"   ❌ Falhas: {final_stats['failed']}")

remaining = len(pdfs_to_process) - len(pdfs_this_session)
if remaining > 0:
    print(f"\n⏭️  Execute novamente para processar os {remaining} PDFs restantes")
    if PDFS_PER_SESSION:
        print(f"   Estimativa: {(remaining // PDFS_PER_SESSION) + 1} sessões adicionais")
else:
    print(f"\n🎉 TODOS OS PDFS FORAM PROCESSADOS!")

print(f"\n📝 Log salvo em: {TRACKING_FILE}")
print(f"{'='*60}")