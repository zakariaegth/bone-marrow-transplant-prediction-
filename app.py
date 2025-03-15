from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Charger le modèle
model = joblib.load('models/model.pkl')

# Route pour la page d'accueil
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Récupérer les données du formulaire
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        clinical_data = float(request.form['clinical_data'])

        # Préparer les données pour la prédiction
        input_data = [[age, sex, clinical_data]]

        # Faire la prédiction
        prediction = model.predict(input_data)[0]

        # Afficher la page de résultat
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)