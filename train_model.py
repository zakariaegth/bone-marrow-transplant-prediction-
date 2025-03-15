from sklearn.ensemble import RandomForestClassifier
import joblib

# Exemple de données d'entraînement (à adapter à votre cas)
X_train = [[1, 0, 50], [2, 1, 60], [3, 0, 70]]  # Features : âge, sexe, donnée clinique
y_train = [1, 0, 1]  # Labels : 1 = succès, 0 = échec

# Créer et entraîner le modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Sauvegarder le modèle dans un fichier .pkl
joblib.dump(model, 'models/model.pkl')
print("Modèle entraîné et sauvegardé dans 'models/model.pkl'")