import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pandas as pd
import numpy as np
import re
import os

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Tweets para Aerolíneas",
    page_icon="✈️",
    layout="wide"
)

# Función para cargar el modelo BERT
@st.cache_resource
def load_bert_model():
    """
    Carga el modelo BERT exportado desde training.ipynb
    """
    # Obtener la ruta absoluta del archivo actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construir la ruta al modelo de forma robusta
    model_path = os.path.join(current_dir, '..', 'code', 'models', 'bert_model.pt')
    model_path = os.path.normpath(model_path)  # Normalizar la ruta

    if not os.path.exists(model_path):
        st.error(f"❌ No se encontró el modelo en: {model_path}")
        st.error("🔧 Ejecuta primero training.ipynb para entrenar y exportar el modelo")
        return None
    
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Cargar estado del modelo
        model_state = torch.load(model_path, map_location='cpu')
        
        # Reconstruir tokenizer primero
        tokenizer = AutoTokenizer.from_pretrained(model_state['tokenizer_name'])
        
        # Reconstruir modelo usando el tokenizer y configuración guardada
        model = AutoModelForSequenceClassification.from_pretrained(
            model_state['tokenizer_name'],
            num_labels=model_state['num_labels'],
            label2id=model_state['label_to_id'],
            id2label=model_state['id_to_label']
        )
        
        # Cargar los pesos entrenados
        model.load_state_dict(model_state['model_state_dict'])
        
        # Configurar dispositivo automáticamente (portable)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'label_to_id': model_state['label_to_id'],
            'id_to_label': model_state['id_to_label'],
            'classes': model_state['label_encoder_classes'],
            'device': device,
            'training_info': model_state.get('training_info', {})
        }
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# Función para preprocesar texto
def preprocess_text(text):
    """
    Preprocesamiento mínimo igual que en el entrenamiento
    """
    # Remover URLs
    text = re.sub(r'http\S+', '', text)
    # Reemplazar mentions
    text = re.sub(r'@\w+', '@user', text)
    return text.strip()

# Función para clasificar tweet
def classify_tweet(text, model_data):
    """
    Clasifica un tweet usando el modelo BERT
    """
    if model_data is None:
        return None
    
    # Preprocesar texto
    text_clean = preprocess_text(text)
    
    # Tokenizar
    inputs = model_data['tokenizer'](
        text_clean,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    ).to(model_data['device'])
    
    # Predecir
    with torch.no_grad():
        outputs = model_data['model'](**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = torch.max(predictions, dim=1)[0].item()
    
    # Convertir id_to_label keys a enteros
    id_to_label = {int(k): v for k, v in model_data['id_to_label'].items()}
    
    # Obtener etiqueta predicha
    predicted_label = id_to_label[predicted_class]
    
    # Obtener todas las probabilidades
    all_probs = {}
    for i, prob in enumerate(predictions[0]):
        if i in id_to_label:
            all_probs[id_to_label[i]] = prob.item()
    
    return {
        'text': text,
        'text_clean': text_clean,
        'predicted_category': predicted_label,
        'predicted_id': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probs
    }

# Cargar modelo
model_data = load_bert_model()

# Interfaz principal
st.title("✈️ Clasificador de Tweets para Aerolíneas")
st.markdown("### Clasifica tweets negativos usando modelo BERT entrenado")

if model_data is not None:
    # Información del modelo
    col1, col2, col3 = st.columns(3)
    
    training_info = model_data.get('training_info', {})
    
    with col1:
        st.metric("Dispositivo", str(model_data['device']).upper())
    
    with col2:
        st.metric("Clases", len(model_data['classes']))
    
    with col3:
        epochs = training_info.get('epochs', 'N/A')
        st.metric("Épocas", epochs)
    
    # Mostrar categorías disponibles
    with st.expander("📋 Categorías disponibles"):
        for i, category in enumerate(model_data['classes']):
            st.write(f"• {category}")
    
    # Área de texto para el tweet
    st.markdown("---")
    tweet = st.text_area(
        "📝 Escribe o pega un tweet negativo:",
        placeholder="Ejemplo: @AmericanAir my flight was delayed 3 hours and my luggage was lost! This is unacceptable!",
        height=100
    )
    
    # Botón de clasificación
    if st.button("🔍 Clasificar Tweet", type="primary"):
        if tweet.strip():
            with st.spinner("🤖 Clasificando..."):
                result = classify_tweet(tweet, model_data)
                
                if result:
                    st.success("✅ Clasificación completada!")
                    
                    # Mostrar resultados principales
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("🎯 Categoría Predicha", result['predicted_category'])
                        st.metric("📊 Confianza", f"{result['confidence']:.1%}")
                    
                    with col2:
                        st.subheader("🔢 ID Predicho")
                        st.write(result['predicted_id'])
                        
                        st.subheader("🧹 Texto Procesado")
                        st.code(result['text_clean'])
                    
                    # Mostrar todas las probabilidades
                    st.subheader("📈 Probabilidades por Categoría")
                    
                    # Crear DataFrame para mejor visualización
                    prob_df = pd.DataFrame([
                        {'Categoría': cat, 'Probabilidad': prob}
                        for cat, prob in result['all_probabilities'].items()
                    ]).sort_values('Probabilidad', ascending=False)
                    
                    # Mostrar como gráfico de barras
                    st.bar_chart(prob_df.set_index('Categoría'))
                    
                    # Mostrar tabla detallada
                    st.dataframe(
                        prob_df.style.format({'Probabilidad': '{:.1%}'}),
                        use_container_width=True
                    )
                else:
                    st.error("❌ Error durante la clasificación")
        else:
            st.warning("⚠️ Por favor, ingresa un tweet para clasificar")

    # Información adicional
    st.markdown("---")
    st.markdown("### ℹ️ Información del Modelo")
    st.markdown(f"**Modelo base:** BERT")
    st.markdown(f"**Preprocesamiento:** Mínimo (URLs y mentions)")
    st.markdown(f"**Máximo de tokens:** 128")
    st.markdown(f"**Dispositivo:** {model_data['device']}")

else:
    st.error("❌ No se pudo cargar el modelo BERT")
    st.info("🔧 Asegúrate de haber ejecutado training.ipynb completamente")
    st.info("📁 El archivo debe estar en: ./models/bert_model.pt")
