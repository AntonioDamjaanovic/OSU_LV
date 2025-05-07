import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Mapa za prikaz klasa
labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

# Funkcija za prikaz granica odluke
def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx],
                    edgecolor='w', label=labels[cl])
    plt.xlabel('bill_length_mm')
    plt.ylabel('flipper_length_mm')
    plt.legend(loc='upper left')
    plt.title('Decision Regions')
    plt.show()

# Učitavanje podataka
df = pd.read_csv('LV5/penguins.csv')

# a) Priprema podataka
df = df.drop(columns=['sex'])  # izbacivanje stupca s puno nedostajućih vrijednosti
df.dropna(axis=0, inplace=True)  # izbacivanje redova s nedostajućim vrijednostima
df['species'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2})

input_vars = ['bill_length_mm', 'flipper_length_mm']
output_var = 'species'

X = df[input_vars].to_numpy()
y = df[output_var].to_numpy()

# Podjela na train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# a) Stupčasti dijagram broja primjera po klasama
train_counts = np.unique(y_train, return_counts=True)[1]
test_counts = np.unique(y_test, return_counts=True)[1]

fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(train_counts))
ax.bar(index, train_counts, bar_width, label='Train', color='deepskyblue')
ax.bar(index + bar_width, test_counts, bar_width, label='Test', color='peachpuff')
ax.set_xlabel('Klasa (Vrsta pingvina)')
ax.set_ylabel('Broj primjera')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['Adelie', 'Chinstrap', 'Gentoo'])
ax.legend()
plt.title('Broj primjera po klasama u train i test skupovima')
plt.show()

# b) Logistička regresija
model = LogisticRegression(multi_class='ovr', max_iter=200)
model.fit(X_train, y_train)

# c) Parametri modela
print("Koeficijenti modela:\n", model.coef_)
print("Presjeci modela:\n", model.intercept_)

# d) Granice odluke
plot_decision_regions(X_train, y_train, model)

# e) Evaluacija na testnom skupu
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Adelie', 'Chinstrap', 'Gentoo'])

print('Matrica zabune:\n', conf_matrix)
print(f'Točnost: {accuracy:.2f}')
print('Classification Report:\n', class_report)

# f) Dodavanje dodatnih ulaznih veličina
input_vars_ext = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']
X_ext = df[input_vars_ext].to_numpy()
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(X_ext, y, test_size=0.2, random_state=123)

model_ext = LogisticRegression(multi_class='ovr', max_iter=200)
model_ext.fit(X_train_ext, y_train_ext)

y_pred_ext = model_ext.predict(X_test_ext)
conf_matrix_ext = confusion_matrix(y_test_ext, y_pred_ext)
accuracy_ext = accuracy_score(y_test_ext, y_pred_ext)
class_report_ext = classification_report(y_test_ext, y_pred_ext, target_names=['Adelie', 'Chinstrap', 'Gentoo'])

print('Matrica zabune s dodatnim značajkama:\n', conf_matrix_ext)
print(f'Točnost s dodatnim značajkama: {accuracy_ext:.2f}')
print('Classification Report s dodatnim značajkama:\n', class_report_ext)
