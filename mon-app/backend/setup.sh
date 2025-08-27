#!/bin/bash
# setup.sh - Script de configuration initiale

echo "🚀 Configuration du backend Détecteur de Plagiat et IA"

# Créer un environnement virtuel
echo "📦 Création de l'environnement virtuel..."
python3 -m venv venv
source venv/bin/activate

# Mettre à jour pip
pip install --upgrade pip

# Installer les dépendances
echo "📚 Installation des dépendances..."
pip install -r requirements.txt

# Télécharger les ressources NLTK
echo "📖 Téléchargement des ressources NLTK..."
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
print('✅ Ressources NLTK téléchargées')
"

# Créer les dossiers nécessaires
echo "📁 Création des dossiers..."
mkdir -p uploads
mkdir -p models
mkdir -p logs
mkdir -p exports

# Initialiser la base de données
echo "🗄️ Initialisation de la base de données..."
python init_db.py

# Configurer les paramètres par défaut
echo "⚙️ Configuration des paramètres par défaut..."
python config_defaults.py

echo "✅ Configuration terminée!"
echo ""
echo "Pour démarrer le serveur:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "L'API sera disponible sur http://localhost:5000"