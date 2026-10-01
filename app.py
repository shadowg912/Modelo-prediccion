# ---------------------------------------------------------
# Aplicación Web: Predictor de Viabilidad de Ideas Tecnológicas
# Regresión Lineal con Interfaz Streamlit
# ---------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Viabilidad de Ideas",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Cargar o entrenar el modelo
# ---------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    """Carga el dataset, entrena el modelo y retorna todo lo necesario."""
    # Cargar datos
    df = pd.read_csv("data.csv")
    
    # Preparar X e y
    X = df.drop(columns=["idea_id", "score_viability"])
    y = df["score_viability"]
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    
    # Evaluar
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Coeficientes
    coef_df = pd.DataFrame({
        "Característica": X.columns,
        "Coeficiente": modelo.coef_
    }).sort_values(by="Coeficiente", ascending=False)
    
    return {
        "df": df,
        "X": X,
        "y": y,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "modelo": modelo,
        "mse": mse,
        "r2": r2,
        "coef_df": coef_df,
        "feature_names": list(X.columns)
    }

# ---------------------------------------------------------
# Análisis de texto para extraer características
# ---------------------------------------------------------
PALABRAS_CLAVE = {
    "market_demand": [
        "demanda", "necesidad", "clientes", "usuarios", "público", "consumidores",
        "interés", "búsqueda", "requerimiento", "solicitud", "mercado objetivo",
        "demandado", "popular", "tendencia", "creciente"
    ],
    "innovation": [
        "innovador", "novedoso", "disruptivo", "único", "revolucionario", "creativo",
        "original", "diferente", "nuevo", "vanguardia", "pionero", "breakthrough",
        "transformador", "disruptión", "innovación"
    ],
    "technical_feasibility": [
        "factible", "viable", "posible", "realizable", "alcanzable", "práctico",
        "robusto", "probado", "testeado", "validado", "funcional", "operativo",
        "tecnología", "desarrollo", "implementación"
    ],
    "market_size": [
        "grande", "amplio", "extenso", "masivo", "global", "enorme", "millones",
        "miles de millones", "escalable", "alcance", "cobertura", "volumen",
        "tamaño", "dimensión", "escala"
    ],
    "competition": [
        "competencia", "competidores", "rivales", "saturado", "diferenciación",
        "ventaja competitiva", "nicho", "exclusivo", "monopolio", "oligopolio",
        "barreras", "competitivo", "mercado copado"
    ],
    "development_cost": [
        "costo", "económico", "barato", "asequible", "bajo costo", "eficiente",
        "optimizado", "reducido", "mínimo", "rentable", "presupuesto", "inversión baja"
    ],
    "profit_margin": [
        "rentable", "beneficio", "ganancia", "margen", "lucrativo", "retorno",
        "ROI", "retorno de inversión", "ganancias", "dividendo", "utilidad"
    ],
    "scalability": [
        "escalable", "crecimiento", "expandir", "ampliar", "multiplicar",
        "replicable", "modular", "flexible", "adaptable", "crecer", "escalar"
    ],
    "investment_required": [
        "inversión", "capital", "financiamiento", "funding", "recursos",
        "presupuesto", "fondos", "capital semilla", "venture", "inversores"
    ],
    "target_market": [
        "mercado objetivo", "público objetivo", "nicho", "segmento", "audiencia",
        "clientes ideales", "target", "demográfico", "vertical", "industria"
    ],
    "growth_potential": [
        "crecimiento", "potencial", "expansión", "proyección", "futuro",
        "escalable", "progresivo", "acelerado", "boom", "explosión", "auge"
    ],
    "customer_problem": [
        "problema", "dolor", "necesidad", "solución", "resolver", "mejorar",
        "optimizar", "simplificar", "facilitar", "ahorrar", "eficiencia"
    ]
}

def analizar_texto(texto: str) -> dict:
    """
    Analiza el texto de una idea y extrae características basadas en palabras clave.
    Retorna un diccionario con valores del 1 al 10 para cada característica.
    """
    texto_lower = texto.lower()
    palabras = texto_lower.split()
    total_palabras = len(palabras)
    
    resultado = {}
    detalle = {}
    
    for caracteristicas, keywords in PALABRAS_CLAVE.items():
        # Contar coincidencias
        coincidencias = sum(1 for kw in keywords if kw in texto_lower)
        
        # Calcular densidad (coincidencias por cada 100 palabras)
        if total_palabras > 0:
            densidad = (coincidencias / total_palabras) * 100
        else:
            densidad = 0
        
        # Convertir densidad a escala 1-10
        # Más coincidencias = mayor puntuación
        if densidad >= 5:
            valor = 10
        elif densidad >= 3:
            valor = 8
        elif densidad >= 2:
            valor = 6
        elif densidad >= 1:
            valor = 4
        elif coincidencias > 0:
            valor = 2
        else:
            # Si no hay coincidencias, asignar un valor neutral basado en longitud
            valor = 5 if total_palabras > 20 else 3
        
        resultado[caracteristicas] = valor
        detalle[caracteristicas] = {
            "coincidencias": coincidencias,
            "densidad": round(densidad, 2),
            "valor": valor
        }
    
    return resultado, detalle

# ---------------------------------------------------------
# Interfaz de la aplicación
# ---------------------------------------------------------
def main():
    # Título
    st.title("🚀 Predictor de Viabilidad de Ideas Tecnológicas")
    st.markdown("""
    Este sistema utiliza **Regresión Lineal** para predecir qué tan viable y atractiva 
    es una idea tecnológica para inversores, basándose en 12 características clave.
    """)
    
    # Cargar modelo
    with st.spinner("Cargando y entrenando modelo..."):
        datos = cargar_modelo()
    
    # Sidebar con información del modelo
    with st.sidebar:
        st.header("📊 Información del Modelo")
        st.metric("R² (Coef. Determinación)", f"{datos['r2']:.3f}")
        st.metric("RMSE", f"{np.sqrt(datos['mse']):.2f}")
        st.metric("Muestras de entrenamiento", len(datos['X']))
        
        st.markdown("---")
        st.header("📈 Coeficientes del Modelo")
        st.dataframe(datos['coef_df'], use_container_width=True)
        
        st.markdown("---")
        st.info("""
        **¿Cómo funciona?**
        El modelo calcula: 
        `score = Σ(coeficiente × valor_característica) + intercepto`
        """)
    
    # Tabs para diferentes modos
    tab1, tab2, tab3 = st.tabs([
        "📝 Modo Manual", 
        "💬 Modo Texto", 
        "📊 Análisis del Modelo"
    ])
    
    # ==========================================
    # TAB 1: MODO MANUAL
    # ==========================================
    with tab1:
        st.header("Ingresa las características de tu idea")
        
        col1, col2 = st.columns(2)
        
        with col1:
            market_demand = st.slider("Market Demand (Demanda del mercado)", 1, 10, 5)
            innovation = st.slider("Innovation (Innovación)", 1, 10, 5)
            technical_feasibility = st.slider("Technical Feasibility (Viabilidad técnica)", 1, 10, 5)
            market_size = st.slider("Market Size (Tamaño del mercado)", 1, 10, 5)
            competition = st.slider("Competition (Competencia)", 1, 10, 5)
            development_cost = st.slider("Development Cost (Costo de desarrollo)", 1, 10, 5)
        
        with col2:
            profit_margin = st.slider("Profit Margin (Margen de beneficio)", 1, 10, 5)
            scalability = st.slider("Scalability (Escalabilidad)", 1, 10, 5)
            investment_required = st.slider("Investment Required (Inversión requerida)", 1, 10, 5)
            target_market = st.slider("Target Market (Mercado objetivo)", 1, 10, 5)
            growth_potential = st.slider("Growth Potential (Potencial de crecimiento)", 1, 10, 5)
            customer_problem = st.slider("Customer Problem (Problema del cliente)", 1, 10, 5)
        
        # Crear diccionario de la idea
        idea_manual = {
            "market_demand": market_demand,
            "innovation": innovation,
            "technical_feasibility": technical_feasibility,
            "market_size": market_size,
            "competition": competition,
            "development_cost": development_cost,
            "profit_margin": profit_margin,
            "scalability": scalability,
            "investment_required": investment_required,
            "target_market": target_market,
            "growth_potential": growth_potential,
            "customer_problem": customer_problem
        }
        
        if st.button("🔮 Predecir Viabilidad", type="primary", use_container_width=True):
            mostrar_prediccion(idea_manual, datos)
    
    # ==========================================
    # TAB 2: MODO TEXTO
    # ==========================================
    with tab2:
        st.header("Describe tu idea en texto libre")
        st.markdown("""
        Escribe una descripción de tu idea tecnológica. El sistema analizará el texto 
        y extraerá automáticamente las características relevantes.
        """)
        
        texto_ejemplo = """Ejemplo: "Nuestra plataforma innovadora utiliza inteligencia artificial 
        para resolver el problema de la gestión de inventarios en pequeñas empresas. 
        El mercado objetivo es amplio con millones de usuarios potenciales. 
        La solución es escalable y tiene un alto potencial de crecimiento. 
        La competencia es limitada y nuestra ventaja competitiva es la tecnología única."""
        
        texto_idea = st.text_area(
            "Describe tu idea:",
            height=200,
            placeholder=texto_ejemplo
        )
        
        if st.button("🔮 Analizar Texto y Predecir", type="primary", use_container_width=True):
            if len(texto_idea.strip()) < 20:
                st.warning("Por favor, escribe al menos 20 caracteres para analizar tu idea.")
            else:
                # Analizar texto
                with st.spinner("Analizando texto..."):
                    caracteristicas, detalle = analizar_texto(texto_idea)
                
                # Mostrar análisis del texto
                st.subheader("📋 Análisis del Texto")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Características extraídas:**")
                    df_detalle = pd.DataFrame([
                        {
                            "Característica": k.replace("_", " ").title(),
                            "Coincidencias": v["coincidencias"],
                            "Densidad": f"{v['densidad']}%",
                            "Valor (1-10)": v["valor"]
                        }
                        for k, v in detalle.items()
                    ])
                    st.dataframe(df_detalle, use_container_width=True)
                
                with col2:
                    st.markdown("**Valores asignados:**")
                    # Mostrar como barras
                    fig, ax = plt.subplots(figsize=(10, 6))
                    nombres = [k.replace("_", " ").title() for k in detalle.keys()]
                    valores = [v["valor"] for v in detalle.values()]
                    colores = plt.cm.RdYlGn(np.array(valores) / 10)
                    
                    ax.barh(nombres, valores, color=colores)
                    ax.set_xlim(0, 10)
                    ax.set_xlabel("Valor (1-10)")
                    ax.set_title("Características extraídas del texto")
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Mostrar predicción
                st.markdown("---")
                mostrar_prediccion(caracteristicas, datos)
    
    # ==========================================
    # TAB 3: ANÁLISIS DEL MODELO
    # ==========================================
    with tab3:
        st.header("Análisis Detallado del Modelo")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² (Coef. Determinación)", f"{datos['r2']:.3f}")
        with col2:
            st.metric("MSE", f"{datos['mse']:.2f}")
        with col3:
            st.metric("RMSE", f"{np.sqrt(datos['mse']):.2f}")
        
        # Gráfico Real vs Predicho
        st.subheader("📈 Real vs Predicho")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(datos['y_test'], datos['y_pred'], alpha=0.7, edgecolors='black')
        ax.plot([datos['y'].min(), datos['y'].max()], 
                [datos['y'].min(), datos['y'].max()], 
                'r--', linewidth=2, label='Línea perfecta')
        ax.set_xlabel("Viabilidad Real")
        ax.set_ylabel("Viabilidad Predicha")
        ax.set_title("Regresión Lineal: Valores Reales vs Predichos")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Importancia de características
        st.subheader("📊 Importancia de Características (Coeficientes)")
        fig, ax = plt.subplots(figsize=(10, 6))
        colores = ['green' if c > 0 else 'red' for c in datos['coef_df']['Coeficiente']]
        ax.barh(datos['coef_df']['Característica'], datos['coef_df']['Coeficiente'], color=colores)
        ax.set_xlabel("Coeficiente")
        ax.set_title("Impacto de cada característica en la viabilidad")
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Distribución de viabilidad
        st.subheader("📉 Distribución de Viabilidad en el Dataset")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(datos['y'], bins=15, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(datos['y'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {datos["y"].mean():.1f}')
        ax.set_xlabel("Score de Viabilidad")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de Viabilidad en las 50 Ideas")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        # Dataset completo
        st.subheader("📋 Dataset Completo")
        st.dataframe(datos['df'], use_container_width=True)

# ---------------------------------------------------------
# Función para mostrar la predicción con cálculo detallado
# ---------------------------------------------------------
def mostrar_prediccion(idea: dict, datos: dict):
    """Muestra la predicción con el cálculo detallado paso a paso."""
    
    # Realizar predicción
    entrada = pd.DataFrame([idea])
    prediccion = datos['modelo'].predict(entrada)[0]
    
    # Determinar nivel
    if prediccion >= 85:
        nivel = "ALTA"
        color = "green"
        emoji = "🟢"
    elif prediccion >= 70:
        nivel = "MEDIA"
        color = "orange"
        emoji = "🟡"
    else:
        nivel = "BAJA"
        color = "red"
        emoji = "🔴"
    
    # Mostrar resultado principal
    st.markdown("---")
    st.subheader("🎯 Resultado de la Predicción")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.metric("Score de Viabilidad", f"{prediccion:.2f}")
    
    with col2:
        st.metric("Nivel", f"{emoji} {nivel}")
    
    with col3:
        # Calcular percentil
        percentil = (datos['y'] < prediccion).mean() * 100
        st.metric("Percentil", f"{percentil:.1f}%")
    
    # Barra de progreso visual
    st.progress(min(prediccion / 100, 1.0))
    
    # ==========================================
    # CÁLCULO DETALLADO
    # ==========================================
    st.markdown("---")
    st.subheader("🧮 Cálculo Detallado Paso a Paso")
    
    st.markdown("""
    La regresión lineal calcula el score usando la fórmula:
    
    **Score = intercepto + Σ(coeficienteᵢ × valorᵢ)**
    """)
    
    # Mostrar la fórmula con valores reales
    intercepto = datos['modelo'].intercept_
    coeficientes = dict(zip(datos['feature_names'], datos['modelo'].coef_))
    
    # Crear tabla de cálculo
    calculo = []
    suma_parcial = intercepto
    
    # Intercepto
    calculo.append({
        "Componente": "Intercepto (β₀)",
        "Coeficiente": f"{intercepto:.4f}",
        "Valor": "-",
        "Contribución": f"{intercepto:.4f}",
        "Acumulado": f"{suma_parcial:.4f}"
    })
    
    # Cada característica
    for feature in datos['feature_names']:
        coef = coeficientes[feature]
        valor = idea[feature]
        contribucion = coef * valor
        suma_parcial += contribucion
        
        calculo.append({
            "Componente": feature.replace("_", " ").title(),
            "Coeficiente": f"{coef:.4f}",
            "Valor": f"{valor}",
            "Contribución": f"{contribucion:.4f}",
            "Acumulado": f"{suma_parcial:.4f}"
        })
    
    # Mostrar tabla
    df_calculo = pd.DataFrame(calculo)
    st.dataframe(df_calculo, use_container_width=True)
    
    # Verificación
    st.success(f"✅ **Verificación:** La suma de todas las contribuciones es **{suma_parcial:.2f}**, que coincide con la predicción del modelo.")
    
    # ==========================================
    # VISUALIZACIÓN DE CONTRIBUCIONES
    # ==========================================
    st.markdown("---")
    st.subheader("📊 Contribución de cada Factor al Resultado")
    
    # Gráfico de contribuciones
    fig, ax = plt.subplots(figsize=(12, 6))
    
    nombres = [c["Componente"] for c in calculo[1:]]  # Excluir intercepto
    contribuciones = [float(c["Contribución"]) for c in calculo[1:]]
    
    colores = ['green' if c > 0 else 'red' for c in contribuciones]
    
    ax.barh(nombres, contribuciones, color=colores, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Contribución al Score")
    ax.set_title("¿Cómo afecta cada característica a la viabilidad?")
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Añadir valores en las barras
    for i, (nombre, contrib) in enumerate(zip(nombres, contribuciones)):
        ax.text(contrib, i, f' {contrib:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ==========================================
    # COMPARACIÓN CON EL DATASET
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Comparación con el Dataset")
    
    # Calcular estadísticas
    media_dataset = datos['y'].mean()
    std_dataset = datos['y'].std()
    diferencia = prediccion - media_dataset
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Media del Dataset", f"{media_dataset:.2f}")
    
    with col2:
        st.metric("Tu Idea", f"{prediccion:.2f}")
    
    with col3:
        delta_color = "normal" if diferencia >= 0 else "inverse"
        st.metric("Diferencia", f"{diferencia:+.2f}", delta_color=delta_color)
    
    # Interpretación
    st.info(f"""
    **Interpretación:** Tu idea tiene un score de **{prediccion:.2f}**, lo que la sitúa 
    en el **percentil {(datos['y'] < prediccion).mean() * 100:.1f}%** del dataset.
    Esto significa que es mejor que el **{(datos['y'] < prediccion).mean() * 100:.1f}%** 
    de las ideas analizadas.
    """)
    
    # Recomendaciones
    st.markdown("---")
    st.subheader("💡 Recomendaciones")
    
    # Identificar características con mayor potencial de mejora
    oportunidades = []
    for feature in datos['feature_names']:
        coef = coeficientes[feature]
        valor_actual = idea[feature]
        
        # Si el coeficiente es positivo y el valor es bajo, hay oportunidad
        if coef > 0 and valor_actual < 7:
            potencial = (10 - valor_actual) * coef
            oportunidades.append({
                "Característica": feature.replace("_", " ").title(),
                "Valor Actual": valor_actual,
                "Coeficiente": f"{coef:.4f}",
                "Potencial de Mejora": f"+{potencial:.2f} puntos"
            })
        # Si el coeficiente es negativo y el valor es alto, hay oportunidad
        elif coef < 0 and valor_actual > 4:
            potencial = (valor_actual - 1) * abs(coef)
            oportunidades.append({
                "Característica": feature.replace("_", " ").title(),
                "Valor Actual": valor_actual,
                "Coeficiente": f"{coef:.4f}",
                "Potencial de Mejora": f"+{potencial:.2f} puntos"
            })
    
    if oportunidades:
        df_oportunidades = pd.DataFrame(oportunidades)
        df_oportunidades = df_oportunidades.sort_values(
            by="Potencial de Mejora", ascending=False
        )
        st.markdown("**Características que podrías mejorar para aumentar la viabilidad:**")
        st.dataframe(df_oportunidades, use_container_width=True)
    else:
        st.success("¡Excelente! Tu idea está bien optimizada según el modelo.")

# ---------------------------------------------------------
# Ejecutar la aplicación
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
