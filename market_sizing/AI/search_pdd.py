import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ========= CONFIGURAÇÕES =========
load_dotenv()

INDEX_NAME = "financial-reports-2"
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# Variações de nomenclatura de PDD 
PDD_QUERIES = [
    "Provisão para Devedores Duvidosos",
    "Provisão/Reversão de Créds. Liquidação Duvidosa",
    "Provisão para Créditos de Liquidação Duvidosa",
    "Provisão ou (reversão) para perdar de créditos com liquidação duvidosa",
    "PCLD",
    "PDD",
    "Perdas estimadas com créditos de liquidação duvidosa",
    "Provisão para perdas de créditos esperadas",
    "Redução ao valor recuperável de contas a receber",
    "Impairment de contas a receber",
]

# ========== CHAVES =========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys não definidas")

# ========== INICIALIZAÇÃO ==========
print("🔍 Sistema de Extração de PDD")
print("=" * 60)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# Verifica estatísticas do índice
stats = index.describe_index_stats()
print(f"📊 Vetores no índice: {stats.total_vector_count}")
print(f"📁 Namespaces: {stats.namespaces}")

# ========== FUNÇÕES DE BUSCA ==========
def search_pdd_multi_query(vectorstore, company: str, year: int, k: int = 20) -> List:
    """Busca usando múltiplas queries para maximizar recall"""
    all_results = []
    seen_content = set()
    
    for query in PDD_QUERIES[:5]:  # Usa as 5 principais variações
        query_with_year = f"{query} {year}"
        
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": 0.65,
                "filter": {
                    "company": company,
                    "is_financial_section": True
                }
            }
        )
        
        try:
            results = retriever.get_relevant_documents(query_with_year)
            
            for doc in results:
                content_hash = hash(doc.page_content[:100])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_results.append(doc)
        except Exception as e:
            print(f"  ⚠️ Erro na query '{query}': {e}")
            continue
    
    return all_results

# ========== PROMPT ESPECIALIZADO ==========
PROMPT_TEMPLATE = """Você é um agente de inteligência artificial especializado em análise de demonstrações financeiras de empresas brasileiras, com expertise em identificar informações sobre a Provisão para Devedores Duvidosos (PDD).

==================================================
CONTEXTO DOS DOCUMENTOS DA EMPRESA {company}:
==================================================
{context}

==================================================
TAREFA PRINCIPAL:
==================================================
Identifique o valor da Provisão para Devedores Duvidosos (PDD) ou Provisão para Créditos de Liquidação Duvidosa (PCLD) para o ano de {year}.

Você deve encontrar especificamente:
1. **SALDO DA PDD NO BALANÇO PATRIMONIAL CONSOLIDADO DE {year}** (estoque no final do período)
   - Localização típica: Ativo Circulante → Contas a Receber → Provisão (valor negativo)
   
2. **DESPESA/REVERSÃO DE PDD NA DRE CONSOLIDADA DE {year}** (fluxo/movimento do período)
   - Representa o quanto foi constituído ou revertido no resultado do ano

==================================================
NOMENCLATURAS POSSÍVEIS DA PDD:
==================================================
A PDD pode aparecer com diversos nomes:
- "Provisão para Devedores Duvidosos"
- "Provisão para Créditos de Liquidação Duvidosa" (PCLD)
- "Provisão para perdas associadas ao risco de crédito"
- "Perdas esperadas com operações de crédito"
- "Perda esperada para crédito de liquidação duvidosa"
- "Allowance for doubtful accounts" (em inglês)
- "Provisão para perdas em operações de crédito"
- "Redução ao valor recuperável de contas a receber"
- "Estimativa de perda com clientes"
- "Impairment de contas a receber"

==================================================
INSTRUÇÕES CRÍTICAS:
==================================================
1. **CONSOLIDADO APENAS**: Retorne SOMENTE valores consolidados
   - Ignore colunas individuais como "Controladora", "Banco", "Segmento Individual"
   
2. **ANO ESPECÍFICO**: Considere APENAS o exercício de {year}
   - Se houver comparativos com anos anteriores, pegue só {year}
   
3. **LOCALIZAÇÃO DOS VALORES**:
   - Balanço Patrimonial: Seção "Ativo Circulante" → "Contas a Receber"
   - DRE: Seção "Despesas Operacionais" ou "Resultado Financeiro"
   - Notas Explicativas: Geralmente nota sobre "Instrumentos Financeiros" ou "Contas a Receber"

4. **FORMATO DOS VALORES**:
   - PDD geralmente aparece como valor NEGATIVO (reduzindo contas a receber) ou entre parênteses
   - Se houver movimentação (saldo inicial, adições, reversões, baixas), busque o SALDO FINAL em {year}
   - Preste atenção nas UNIDADES: pode estar em R$ (unidade), R$ mil ou R$ milhões

5. **NÃO INVENTE NÚMEROS**: 
   - Se não encontrar o valor específico de {year} consolidado, diga claramente "PDD não identificada"
   - Seja honesto sobre a confiança da informação encontrada

==================================================
FORMATO DA RESPOSTA (OBRIGATÓRIO):
==================================================

**SALDO NO BALANÇO PATRIMONIAL {year}:**
Valor: [número com unidade, ex: R$ 5.234 mil] ou [Não identificado]
Localização: [Balanço Patrimonial - Ativo Circulante / Nota Explicativa X, Página Y]
Nomenclatura: [nome exato encontrado no documento]

**DESPESA/REVERSÃO NA DRE {year}:**
Valor: [número com unidade, ex: R$ 1.200 mil] ou [Não identificado]
Localização: [DRE / Nota Explicativa X, Página Y]
Tipo: [Constituição/Reversão/Complemento]

**CONFIANÇA:** [Alta/Média/Baixa]

**OBSERVAÇÕES:** [Qualquer informação adicional relevante, como contexto sobre movimentações ou metodologia de cálculo mencionada]

==================================================
EXEMPLO DE RESPOSTA IDEAL:
==================================================

**SALDO NO BALANÇO PATRIMONIAL 2024:**
Valor: R$ 15.234 mil
Localização: Balanço Patrimonial Consolidado - Ativo Circulante, Nota Explicativa 8, Página 45
Nomenclatura: Provisão para Créditos de Liquidação Duvidosa (PCLD)

**DESPESA/REVERSÃO NA DRE 2024:**
Valor: R$ 3.450 mil
Localização: DRE Consolidada - Despesas Operacionais, Nota Explicativa 8, Página 48
Tipo: Constituição de provisão

**CONFIANÇA:** Alta

**OBSERVAÇÕES:** A empresa utiliza modelo de perda esperada (IFRS 9). Houve aumento de 28% na provisão devido à deterioração da carteira de clientes no setor de varejo.

==================================================
PERGUNTA:
==================================================
Qual o valor da PDD (saldo no balanço e movimentação na DRE) para o ano de {year}?

RESPOSTA:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "company", "year"]
)

# ========== LLM COM STRUCTURED OUTPUT ==========
llm = ChatOpenAI(
    model="gpt-4o",  # Ou gpt-4o-mini para economizar
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 25}),
    chain_type_kwargs={"prompt": PROMPT}
)

# ========== VALIDAÇÃO COM REGEX ==========
def validate_pdd_value(text: str) -> Optional[str]:
    """Extrai valor monetário da resposta"""
    patterns = [
        r'R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*(?:mil|milhões|MM)?',
        r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?)\s*(?:mil|milhões|MM)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None

def parse_confidence(text: str) -> str:
    """Extrai nível de confiança"""
    text_lower = text.lower()
    if 'confiança: alta' in text_lower or 'alta confiança' in text_lower:
        return "Alta"
    elif 'confiança: média' in text_lower or 'média confiança' in text_lower:
        return "Média"
    elif 'confiança: baixa' in text_lower or 'baixa confiança' in text_lower:
        return "Baixa"
    return "Desconhecida"

# ========== FUNÇÃO PRINCIPAL DE EXTRAÇÃO ==========
def extract_pdd(company: str, year: int = 2024, verbose: bool = True) -> Dict:
    """Extrai PDD de uma empresa específica"""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🏢 Empresa: {company}")
        print(f"📅 Ano: {year}")
        print(f"{'='*60}")
    
    # Busca multi-query
    relevant_docs = search_pdd_multi_query(vectorstore, company, year, k=20)
    
    if not relevant_docs:
        print(f"⚠️  Nenhum documento relevante encontrado para {company}")
        return {
            "company": company,
            "year": year,
            "status": "no_documents",
            "pdd_value": None,
            "confidence": None,
            "full_response": None
        }
    
    if verbose:
        print(f"📊 Encontrados {len(relevant_docs)} trechos relevantes")
        print(f"\nPreview dos 2 trechos mais relevantes:")
        for i, doc in enumerate(relevant_docs[:2], 1):
            print(f"\n--- Trecho {i} (Página {doc.metadata.get('page')}) ---")
            preview = doc.page_content[:250].replace('\n', ' ')
            print(f"{preview}...")
    
    # Monta contexto
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs[:15]])
    
    # Consulta o LLM
    try:
        result = qa_chain.invoke({
            "query": f"Qual o valor da PDD para {year}?",
            "context": context,
            "company": company,
            "year": year
        })
        
        llm_response = result['result']
        
        if verbose:
            print(f"\n{'='*60}")
            print("🤖 RESPOSTA DO LLM:")
            print(f"{'='*60}")
            print(llm_response)
            print(f"{'='*60}")
        
        # Valida e extrai informações
        pdd_value = validate_pdd_value(llm_response)
        confidence = parse_confidence(llm_response)
        
        # Verifica se encontrou
        if "não identificada" in llm_response.lower() or not pdd_value:
            status = "not_found"
        else:
            status = "found"
        
        return {
            "company": company,
            "year": year,
            "status": status,
            "pdd_value": pdd_value,
            "confidence": confidence,
            "full_response": llm_response,
            "num_docs_retrieved": len(relevant_docs),
            "extracted_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Erro ao processar {company}: {e}")
        return {
            "company": company,
            "year": year,
            "status": "error",
            "error": str(e),
            "pdd_value": None,
            "confidence": None
        }

# ========== EXTRAÇÃO EM LOTE ==========
def extract_all_companies(year: int = 2024) -> List[Dict]:
    """Extrai PDD de todas as empresas indexadas"""
    
    # Lista empresas únicas no índice
    # (Pinecone não tem API direta, então fazemos query ampla)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1000})
    sample_docs = retriever.get_relevant_documents("empresa demonstrativo financeiro")
    
    companies = set()
    for doc in sample_docs:
        company = doc.metadata.get("company")
        if company:
            companies.add(company)
    
    companies = sorted(list(companies))
    print(f"\n📋 Empresas encontradas: {len(companies)}")
    print(f"   {', '.join(companies[:5])}..." if len(companies) > 5 else f"   {', '.join(companies)}")
    
    results = []
    for i, company in enumerate(companies, 1):
        print(f"\n[{i}/{len(companies)}] Processando {company}...")
        result = extract_pdd(company, year, verbose=False)
        results.append(result)
        
        # Preview
        if result['status'] == 'found':
            print(f"  ✅ PDD: {result['pdd_value']} (Confiança: {result['confidence']})")
        elif result['status'] == 'not_found':
            print(f"  ⚠️  PDD não identificada")
        else:
            print(f"  ❌ Erro: {result.get('error', 'Desconhecido')}")
    
    return results

# ========== SALVAR RESULTADOS ==========
def save_results(results: List[Dict], filename: str = None):
    """Salva resultados em JSON"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pdd_extraction_{timestamp}.json"
    
    filepath = RESULTS_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados salvos em: {filepath}")
    return filepath

# ========== GERAR RELATÓRIO ==========
def generate_report(results: List[Dict]):
    """Gera relatório resumido"""
    total = len(results)
    found = sum(1 for r in results if r['status'] == 'found')
    not_found = sum(1 for r in results if r['status'] == 'not_found')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    high_conf = sum(1 for r in results if r.get('confidence') == 'Alta')
    med_conf = sum(1 for r in results if r.get('confidence') == 'Média')
    low_conf = sum(1 for r in results if r.get('confidence') == 'Baixa')
    
    print(f"\n{'='*60}")
    print("📊 RELATÓRIO FINAL")
    print(f"{'='*60}")
    print(f"Total de empresas: {total}")
    print(f"  ✅ PDD encontrada: {found} ({found/total*100:.1f}%)")
    print(f"  ⚠️  PDD não identificada: {not_found} ({not_found/total*100:.1f}%)")
    print(f"  ❌ Erros: {errors}")
    print(f"\nDistribuição de confiança:")
    print(f"  🟢 Alta: {high_conf}")
    print(f"  🟡 Média: {med_conf}")
    print(f"  🔴 Baixa: {low_conf}")
    print(f"{'='*60}")

# ========== MODO DE USO ==========
if __name__ == "__main__":
    import sys
    
    # Uso: python search_pdd.py [company_name] [year]
    # Ou: python search_pdd.py --all [year]
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
            results = extract_all_companies(year)
            generate_report(results)
            save_results(results)
        else:
            company = sys.argv[1]
            year = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
            result = extract_pdd(company, year, verbose=True)
            save_results([result], f"pdd_{company}_{year}.json")
    else:
        print("\n💡 Como usar:")
        print("  python search_pdd.py '3R PETROLEUM' 2024")
        print("  python search_pdd.py --all 2024")
        print("\nExemplo interativo:")
        result = extract_pdd("3R PETROLEUM", 2024, verbose=True)