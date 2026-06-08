import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# 1. Configuração do Descritor Clássico (ORB)
orb = cv2.ORB_create(nfeatures=500)

# Criamos o objeto que vai comparar os pontos das duas imagens
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def extract_features(image_crop):
    """Extrai pontos-chave e descritores usando ORB."""
    # ORB funciona melhor em tons de cinza
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return descriptors

# 2. Geração de Propostas de Regiões
def generate_region_proposals(image, max_proposals=100):
    """Usa Selective Search para gerar caixas candidatas."""
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects[:max_proposals]

# 3. Cálculo de Posição Geográfica (IoU)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    if float(boxAArea + boxBArea - interArea) == 0: return 0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# 4. Remoção de Duplicatas (NMS)
def remover_duplicatas_nms(resultados, limite_iou=0.3):
    """
    Filtra o ranking para remover caixas sobrepostas no mesmo documento.
    Mantém apenas a caixa de maior score para cada área.
    """
    resultados_limpos = []
    for atual in resultados:
        duplicata = False
        for aprovado in resultados_limpos:
            if atual['doc_name'] == aprovado['doc_name']:
                sobreposicao = calculate_iou(atual['box'], aprovado['box'])
                if sobreposicao > limite_iou:
                    duplicata = True
                    break
        if not duplicata:
            resultados_limpos.append(atual)
    return resultados_limpos

# 5. Indexação (Construção do Banco de Dados)
def build_index(document_paths):
    database = []
    print(f"Indexando {len(document_paths)} documentos...")
    
    for doc_path in document_paths:
        img = cv2.imread(doc_path)
        if img is None: continue
        
        proposals = generate_region_proposals(img, max_proposals=50)
        
        for (x, y, w, h) in proposals:
            if w < 20 or h < 20: continue 
            
            crop = img[y:y+h, x:x+w]
            descriptors = extract_features(crop)
            
            # Só guarda a região se o ORB encontrou pontos úteis nela
            if descriptors is not None:
                database.append({
                    'doc_name': os.path.basename(doc_path),
                    'box': [x, y, w, h],
                    'descriptors': descriptors
                })
            
    return database

# 6. Ranqueamento e Recuperação (Busca)
def retrieve_relevant_regions(query_img_path, expected_box, database, alpha=0.85):
    query_img = cv2.imread(query_img_path)
    query_descriptors = extract_features(query_img)
    
    results = []
    
    if query_descriptors is None:
        print("Erro: Nenhum ponto-chave encontrado na imagem de busca.")
        return []

    for item in database:
        db_descriptors = item['descriptors']
        
        # Faz o "match" bruto entre os pontos da query e do banco
        raw_matches = matcher.match(query_descriptors, db_descriptors)
        
        # --- OPÇÃO 2: FILTRO DE QUALIDADE RIGOROSO ---
        # Só aceita os pontos que casarem com muita perfeição (distância < 50)
        matches = [m for m in raw_matches if m.distance < 50]
        
        # Calcula score visual baseado na quantidade e qualidade dos matches VÁLIDOS
        if len(matches) > 0:
            media_distancia = sum([m.distance for m in matches]) / len(matches)
            vis_sim = max(0, (100 - media_distancia) / 100) 
            fator_quantidade = min(len(matches) / 50, 1.0)
            vis_sim = vis_sim * fator_quantidade
        else:
            vis_sim = 0
        
        spatial_sim = calculate_iou(expected_box, item['box'])
        final_score = (alpha * vis_sim) + ((1 - alpha) * spatial_sim)
        
        results.append({
            'doc_name': item['doc_name'],
            'box': item['box'],
            'score': final_score,
            'vis_sim': vis_sim,
            'iou': spatial_sim,
            'matches_count': len(matches)
        })
        
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    results_sem_duplicatas = remover_duplicatas_nms(results)
    
    return results_sem_duplicatas[:10]

# --- EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    doc_files = glob.glob(r'C:\Users\guiev\Downloads\archive\signatures\full_org\*.png')[:25]
    
    db_index = build_index(doc_files)
    print(f"Índice criado com {len(db_index)} regiões candidatas válidas.")
    
    query_files = glob.glob(r'C:\Users\guiev\Downloads\archive\signatures\full_forg\*.png')[:5]
    print(f"Assinaturas de busca encontradas: {len(query_files)}")
    expected_signature_box = [10, 10, 500, 250] 
    
    for i, query_path in enumerate(query_files):
        print(f"\n--- Processando Query {i+1}: {os.path.basename(query_path)} ---")
        
        top_results = retrieve_relevant_regions(query_path, expected_signature_box, db_index, alpha=0.85)
        
        print("Top Resultados:")
        for rank, res in enumerate(top_results):
            print(f"{rank+1}. Documento: {res['doc_name']} | BBox: {res['box']} | "
                  f"Score Final: {res['score']:.4f} (Visual: {res['vis_sim']:.4f}, IoU: {res['iou']:.4f}, Matches: {res['matches_count']})")
        
        # PARTE VISUAL
        if top_results:
            melhor_resultado = top_results[0]
            nome_documento = melhor_resultado['doc_name']
            
            caminho_doc = rf'C:\Users\guiev\Downloads\archive\signatures\full_org\{nome_documento}'
            img_doc = cv2.imread(caminho_doc)
            
            if img_doc is not None:
                # 1. Pega a caixa e desenha no documento original
                x, y, w, h = melhor_resultado['box']
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(img_doc, (x, y), (x + w, y + h), (0, 0, 255), 4) 
                
                # 2. Converte as imagens para RGB para exibir corretamente
                query_img_rgb = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
                img_doc_rgb = cv2.cvtColor(img_doc, cv2.COLOR_BGR2RGB)
                
                # 3. Cria a figura lado a lado
                plt.figure(figsize=(15, 6))
                
                # Plot 1: A Falsificação (Query)
                plt.subplot(1, 2, 1)
                plt.imshow(query_img_rgb)
                plt.title(f"Consulta (Falsificação): {os.path.basename(query_path)}")
                plt.axis('off')
                
                # Plot 2: O Documento Encontrado (Resultado)
                plt.subplot(1, 2, 2)
                plt.imshow(img_doc_rgb)
                plt.title(f"Resultado Top 1: {nome_documento}\n(Matches Qualificados: {melhor_resultado['matches_count']})")
                plt.axis('off')
                
                # 4. Salva a imagem final pronta para o relatório
                nome_salvar = f"relatorio_query_{i+1}.png"
                plt.savefig(nome_salvar, bbox_inches='tight')
                plt.close()
                print(f"-> Imagem para o relatório salva como {nome_salvar}\n")