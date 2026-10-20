"""
Trabajo Práctico N° 5 — Punto B
Predicción de abandono estudiantil mediante árboles de decisión

Instituto Superior de Formación Técnica N° 197
Alumno: Lucas Nahuel Llanes
Año: 2026
"""

import os
import glob
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

# ============================================================
# CONFIGURACIÓN DE ESTILO PERSONALIZADO
# ============================================================
plt.rcParams.update({
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#adb5bd',
    'axes.labelcolor': '#212529',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.color': '#495057',
    'ytick.color': '#495057',
    'grid.color': '#dee2e6',
    'grid.linestyle': '--',
    'font.family': 'sans-serif',
    'font.size': 10,
})

# Paleta personalizada
COLOR_PRINCIPAL = '#0d6efd'     # azul moderno
COLOR_SECUNDARIO = '#fd7e14'    # naranja
COLOR_ACENTO = '#198754'        # verde

os.chdir(os.path.dirname(os.path.abspath(__file__)))

RANDOM_SEED = 42


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def localizar_planilla(palabra_clave):
    """Devuelve el primer .xlsx de la carpeta que contenga la palabra clave."""
    coincidencias = [f for f in glob.glob("*.xlsx")
                     if palabra_clave.lower() in f.lower()
                     and not f.startswith("predicciones")
                     and not f.startswith("~$")]
    if not coincidencias:
        raise FileNotFoundError(f"No se encontró ningún .xlsx con '{palabra_clave}'")
    print(f"Planilla cargada: {coincidencias[0]}")
    return coincidencias[0]


def limpiar_y_codificar(df):
    """Normaliza y codifica el DataFrame. Se aplica igual a ambos archivos."""
    df = df.rename(columns=RENOMBRE).copy()

    for columna in ["genero", "carrera", "trabaja", "extracurriculares"]:
        df[columna] = df[columna].astype(str).str.strip()
    df["carrera"] = (df["carrera"]
                     .str.replace("TÉCNICATURA", "TECNICATURA")
                     .str.replace("TECNICATURA SUPERIOR EN ", ""))

    df["genero"] = df["genero"].map({"m": 0, "f": 1})
    df["trabaja"] = df["trabaja"].map({"No": 0, "Si": 1})
    df["extracurriculares"] = df["extracurriculares"].map({"No": 0, "Si": 1})

    if "estado_final" in df.columns:
        df["estado_final"] = df["estado_final"].map({"Abandonó": 1, "Continúa": 0})

    return df


# ============================================================
# 1. CARGA Y PREPARACIÓN
# ============================================================

RENOMBRE = {
    "edad": "edad", "genero": "genero", "carrera": "carrera",
    "PromedioPrimerCuatrimestre": "promedio",
    "CantMateriasAprobadasPrimerCuatrimestre": "aprobadas",
    "CantMateriasDesaprobadasPrimerCuatrimestre": "desaprobadas",
    "AsistenciaPromedio(%)": "asistencia",
    "trabaja/NoTrabaja": "trabaja",
    "DistanciaDomicilioAlInstituto(Kms)": "distancia_km",
    "ActividadesExtracurriculares(Estudio)": "extracurriculares",
    "EstadoFinal": "estado_final",
}

planilla_train = localizar_planilla("Entrenamiento")
planilla_predict = localizar_planilla("DatosFinal")

df_train = limpiar_y_codificar(pd.read_excel(planilla_train, sheet_name=0))
df_predict = limpiar_y_codificar(pd.read_excel(planilla_predict, sheet_name=0))

lista_carreras = sorted(set(df_train["carrera"]) | set(df_predict["carrera"]))
df_train["carrera"] = pd.Categorical(df_train["carrera"], categories=lista_carreras)
df_predict["carrera"] = pd.Categorical(df_predict["carrera"], categories=lista_carreras)
df_train = pd.get_dummies(df_train, columns=["carrera"], prefix="carr", dtype=int)
df_predict = pd.get_dummies(df_predict, columns=["carrera"], prefix="carr", dtype=int)

assert df_train.isna().sum().sum() == 0
assert df_predict.isna().sum().sum() == 0

X = df_train.drop(columns="estado_final")
y = df_train["estado_final"]
X_nuevos = df_predict[X.columns]

print("=== DATOS CARGADOS ===")
print(f"Entrenamiento: {len(df_train)} | Predicción: {len(df_predict)}")
print(y.value_counts().to_string(), "\n")


# ============================================================
# 2. DIVISIÓN Y ENTRENAMIENTO
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)

arbol = DecisionTreeClassifier(
    criterion="gini", max_depth=4, min_samples_leaf=5, random_state=RANDOM_SEED)
arbol.fit(X_train, y_train)


# ============================================================
# 3. EVALUACIÓN
# ============================================================

pred_train = arbol.predict(X_train)
pred_test = arbol.predict(X_test)

print("=== EVALUACIÓN ===")
print(f"Accuracy entrenamiento: {accuracy_score(y_train, pred_train):.2f}")
print(f"Accuracy prueba:        {accuracy_score(y_test, pred_test):.2f}\n")

print(pd.DataFrame(confusion_matrix(y_test, pred_test, labels=[0, 1]),
                   index=["Real: Continúa", "Real: Abandonó"],
                   columns=["Pred: Continúa", "Pred: Abandonó"]), "\n")

print(classification_report(y_test, pred_test,
                            target_names=["Continúa", "Abandonó"], zero_division=0))


# ------------------------------------------------------------
# GRÁFICO 1: Matriz de confusión (estilo personalizado)
# ------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 6), facecolor='#f8f9fa')
cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
sns_heatmap = ax1.imshow(cm, cmap='Blues', aspect='auto')

# Agregar valores dentro de cada celda
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else '#212529',
                fontsize=18, fontweight='bold')

ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Continúa', 'Abandonó'])
ax1.set_yticklabels(['Continúa', 'Abandonó'])
ax1.set_xlabel('Predicción del modelo', fontweight='bold')
ax1.set_ylabel('Estado real del estudiante', fontweight='bold')
ax1.set_title('Matriz de confusión — conjunto de prueba', pad=15)

# Barra de color personalizada
cbar = fig1.colorbar(sns_heatmap, ax=ax1)
cbar.set_label('Cantidad de estudiantes', fontweight='bold')

fig1.tight_layout()
fig1.savefig("matriz_confusion.png", dpi=150, facecolor='#f8f9fa')
plt.show()
plt.close(fig1)


# ============================================================
# 4. INTERPRETACIÓN
# ============================================================

importancias = (pd.Series(arbol.feature_importances_, index=X.columns)
                .sort_values(ascending=False))
usadas = importancias[importancias > 0]

print("=== IMPORTANCIA DE VARIABLES ===")
print(usadas.round(3).to_string(), "\n")


# ------------------------------------------------------------
# GRÁFICO 2: Importancia de variables (estilo personalizado)
# ------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='#f8f9fa')

# Colores degradados según importancia
colores = plt.cm.viridis(usadas.sort_values() / usadas.max())

usadas.sort_values().plot.barh(
    color=colores,
    edgecolor='white',
    linewidth=1.5,
    ax=ax2
)

ax2.set_xlabel('Importancia (reducción de impureza Gini)', fontweight='bold')
ax2.set_title('Importancia de las variables en el árbol', pad=15)
ax2.grid(axis='x', alpha=0.3)
ax2.set_axisbelow(True)

# Agregar los valores al lado de cada barra
for i, (nombre, valor) in enumerate(usadas.sort_values().items()):
    ax2.text(valor + 0.005, i, f'{valor:.3f}',
            va='center', fontsize=10, fontweight='bold', color='#212529')

fig2.tight_layout()
fig2.savefig("importancia_variables.png", dpi=150, facecolor='#f8f9fa')
plt.show()
plt.close(fig2)

print("=== REGLAS DEL ÁRBOL ===")
print(export_text(arbol, feature_names=list(X.columns)))


# ------------------------------------------------------------
# GRÁFICO 3: Árbol de decisión (estilo personalizado)
# ------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(24, 12), facecolor='#f8f9fa')
plot_tree(
    arbol,
    feature_names=X.columns,
    class_names=["Continúa", "Abandonó"],
    filled=True,
    rounded=True,
    fontsize=10,
    impurity=True,
    proportion=False,
    ax=ax3,
    precision=2,
)
ax3.set_title('Árbol de decisión — Predicción de abandono',
              fontsize=16, fontweight='bold', pad=20)

fig3.tight_layout()
fig3.savefig("arbol_decision.png", dpi=150, facecolor='#f8f9fa')
plt.show()
plt.close(fig3)


# ============================================================
# 5. ESTABILIDAD
# ============================================================

acc_test = accuracy_score(y_test, pred_test)
acc_trivial = max(y_test.mean(), 1 - y_test.mean())
nodo_raiz = X.columns[arbol.tree_.feature[0]]

print("=== ESTABILIDAD DE LAS VARIABLES ===")
top_por_division = []
for semilla in range(5):
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3,
                                      random_state=semilla, stratify=y)
    arbol_temp = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                        random_state=RANDOM_SEED).fit(Xa, ya)
    top = (pd.Series(arbol_temp.feature_importances_, index=X.columns)
           .sort_values(ascending=False).index[:3].tolist())
    top_por_division.append(top[0])
    print(f"División {semilla}: accuracy = {accuracy_score(yb, arbol_temp.predict(Xb)):.2f} | "
          f"top 3 = {', '.join(top)}")
estable = len(set(top_por_division)) == 1

print("\n=== INTERPRETACIÓN (Punto 4) ===")
print(f"- Variables más importantes: {', '.join(usadas.index[:3])}.")
print(f"- Primer corte: '{nodo_raiz}'.")
print(f"- Accuracy prueba: {acc_test:.2f} | Trivial: {acc_trivial:.2f}")

if acc_test <= acc_trivial:
    print("- El árbol NO supera al modelo trivial.")
else:
    print("- El árbol supera al modelo trivial.")

if estable:
    print(f"- Variable principal estable: '{top_por_division[0]}'.")
else:
    print("- La variable principal CAMBIA según la división.")

if acc_test <= acc_trivial or not estable:
    print("=> Conclusión: no se puede afirmar que alguna variable sea predictiva.\n")
else:
    print("=> Conclusión: las variables parecen relevantes.\n")


# ============================================================
# 6. PREDICCIÓN FINAL
# ============================================================

resultado = pd.read_excel(planilla_predict, sheet_name=0)
resultado["Prediccion"] = ["Abandonó" if p == 1 else "Continúa"
                           for p in arbol.predict(X_nuevos)]
resultado["ProbAbandono"] = arbol.predict_proba(X_nuevos)[:, 1].round(2)
resultado = resultado.sort_values("ProbAbandono", ascending=False)
resultado.to_excel("predicciones_DatosFinal.xlsx", index=False)

print("=== PREDICCIÓN SOBRE DatosFinal ===")
print(resultado["Prediccion"].value_counts().to_string())
print("\nArchivos generados: predicciones_DatosFinal.xlsx, arbol_decision.png, "
      "matriz_confusion.png, importancia_variables.png")