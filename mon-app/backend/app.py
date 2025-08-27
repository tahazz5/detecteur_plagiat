# app.py - API Flask principale
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import uuid
import json

# Imports pour l'analyse
from plagiarism_detector import PlagiarismDetector
from ai_detector import AIDetector
from document_processor import DocumentProcessor

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plagiarism_detector.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

db = SQLAlchemy(app)

# Créer le dossier uploads s'il n'existe pas
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Modèles de base de données
class Submission(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = db.Column(db.String(255))
    student_id = db.Column(db.String(100))
    text_content = db.Column(db.Text)
    plagiarism_score = db.Column(db.Float)
    ai_score = db.Column(db.Float)
    analysis_result = db.Column(db.Text)  # JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class DetectedSource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.String(36), db.ForeignKey('submission.id'))
    source_url = db.Column(db.String(500))
    similarity_score = db.Column(db.Float)
    source_type = db.Column(db.String(100))
    matched_text = db.Column(db.Text)

# Initialisation des détecteurs
plagiarism_detector = PlagiarismDetector()
ai_detector = AIDetector()
document_processor = DocumentProcessor()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Point de santé de l'API"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """Analyser un texte pour plagiat et IA"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        student_id = data.get('student_id', 'anonymous')
        
        if not text:
            return jsonify({'error': 'Texte vide'}), 400
        
        if len(text) < 50:
            return jsonify({'error': 'Texte trop court (minimum 50 caractères)'}), 400
        
        # Analyse du plagiat
        plagiarism_result = plagiarism_detector.detect_plagiarism(text)
        
        # Analyse IA
        ai_result = ai_detector.detect_ai_text(text)
        
        # Compilation des résultats
        analysis_result = {
            'plagiarism': plagiarism_result,
            'ai_detection': ai_result,
            'text_metrics': {
                'word_count': len(text.split()),
                'sentence_count': len([s for s in text.split('.') if s.strip()]),
                'character_count': len(text),
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
            },
            'recommendations': generate_recommendations(
                plagiarism_result['score'], 
                ai_result['score']
            ),
            'analysis_date': datetime.utcnow().isoformat()
        }
        
        # Sauvegarde en base
        submission = Submission(
            student_id=student_id,
            text_content=text,
            plagiarism_score=plagiarism_result['score'],
            ai_score=ai_result['score'],
            analysis_result=json.dumps(analysis_result)
        )
        db.session.add(submission)
        
        # Sauvegarde des sources détectées
        for source in plagiarism_result.get('sources', []):
            detected_source = DetectedSource(
                submission_id=submission.id,
                source_url=source['url'],
                similarity_score=source['similarity'],
                source_type=source['type'],
                matched_text=source.get('matched_text', '')
            )
            db.session.add(detected_source)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'submission_id': submission.id,
            'analysis': analysis_result
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur d\'analyse: {str(e)}'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload et analyse d'un fichier"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        student_id = request.form.get('student_id', 'anonymous')
        
        if file.filename == '':
            return jsonify({'error': 'Nom de fichier vide'}), 400
        
        # Vérifier l'extension
        allowed_extensions = {'.txt', '.docx', '.pdf'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({'error': 'Format de fichier non supporté'}), 400
        
        # Sauvegarder le fichier
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extraire le texte
        text = document_processor.extract_text(filepath)
        
        if not text or len(text.strip()) < 50:
            os.remove(filepath)  # Supprimer le fichier
            return jsonify({'error': 'Impossible d\'extraire le texte ou texte trop court'}), 400
        
        # Analyse (même logique que /analyze)
        plagiarism_result = plagiarism_detector.detect_plagiarism(text)
        ai_result = ai_detector.detect_ai_text(text)
        
        analysis_result = {
            'plagiarism': plagiarism_result,
            'ai_detection': ai_result,
            'text_metrics': {
                'word_count': len(text.split()),
                'sentence_count': len([s for s in text.split('.') if s.strip()]),
                'character_count': len(text),
                'paragraph_count': len([p for p in text.split('\n\n') if p.strip()])
            },
            'recommendations': generate_recommendations(
                plagiarism_result['score'], 
                ai_result['score']
            ),
            'analysis_date': datetime.utcnow().isoformat()
        }
        
        # Sauvegarde
        submission = Submission(
            filename=filename,
            student_id=student_id,
            text_content=text,
            plagiarism_score=plagiarism_result['score'],
            ai_score=ai_result['score'],
            analysis_result=json.dumps(analysis_result)
        )
        db.session.add(submission)
        db.session.commit()
        
        # Nettoyer le fichier uploadé
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'submission_id': submission.id,
            'filename': filename,
            'text_preview': text[:200] + '...',
            'analysis': analysis_result
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur de traitement: {str(e)}'}), 500

@app.route('/api/submissions', methods=['GET'])
def get_submissions():
    """Récupérer l'historique des soumissions"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        student_id = request.args.get('student_id')
        
        query = Submission.query
        if student_id:
            query = query.filter_by(student_id=student_id)
        
        submissions = query.order_by(Submission.created_at.desc()).paginate(
            page=page, per_page=per_page
        )
        
        result = []
        for submission in submissions.items:
            result.append({
                'id': submission.id,
                'filename': submission.filename,
                'student_id': submission.student_id,
                'plagiarism_score': submission.plagiarism_score,
                'ai_score': submission.ai_score,
                'created_at': submission.created_at.isoformat(),
                'text_preview': submission.text_content[:100] + '...' if submission.text_content else None
            })
        
        return jsonify({
            'submissions': result,
            'total': submissions.total,
            'pages': submissions.pages,
            'current_page': page
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur de récupération: {str(e)}'}), 500

@app.route('/api/submissions/<submission_id>', methods=['GET'])
def get_submission_detail(submission_id):
    """Récupérer le détail d'une soumission"""
    try:
        submission = Submission.query.get_or_404(submission_id)
        
        # Récupérer les sources détectées
        sources = DetectedSource.query.filter_by(submission_id=submission_id).all()
        
        result = {
            'id': submission.id,
            'filename': submission.filename,
            'student_id': submission.student_id,
            'text_content': submission.text_content,
            'plagiarism_score': submission.plagiarism_score,
            'ai_score': submission.ai_score,
            'analysis_result': json.loads(submission.analysis_result) if submission.analysis_result else {},
            'created_at': submission.created_at.isoformat(),
            'detected_sources': [{
                'url': source.source_url,
                'similarity': source.similarity_score,
                'type': source.source_type,
                'matched_text': source.matched_text
            } for source in sources]
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Erreur de récupération: {str(e)}'}), 500

@app.route('/api/export/<submission_id>', methods=['GET'])
def export_report(submission_id):
    """Exporter un rapport au format JSON"""
    try:
        submission = Submission.query.get_or_404(submission_id)
        analysis = json.loads(submission.analysis_result) if submission.analysis_result else {}
        
        report = {
            'submission_info': {
                'id': submission.id,
                'filename': submission.filename,
                'student_id': submission.student_id,
                'analysis_date': submission.created_at.isoformat()
            },
            'scores': {
                'plagiarism': submission.plagiarism_score,
                'ai_probability': submission.ai_score
            },
            'detailed_analysis': analysis,
            'text_sample': submission.text_content[:500] + '...' if len(submission.text_content) > 500 else submission.text_content
        }
        
        return jsonify(report)
        
    except Exception as e:
        return jsonify({'error': f'Erreur d\'export: {str(e)}'}), 500

def generate_recommendations(plagiarism_score, ai_score):
    """Générer des recommandations basées sur les scores"""
    recommendations = []
    
    if plagiarism_score > 70:
        recommendations.append({
            'type': 'warning',
            'message': 'Taux de plagiat très élevé - Investigation approfondie recommandée',
            'action': 'Convoquer l\'étudiant pour clarifications'
        })
    elif plagiarism_score > 50:
        recommendations.append({
            'type': 'caution',
            'message': 'Taux de plagiat élevé - Vérification manuelle recommandée',
            'action': 'Examiner les sources identifiées'
        })
    elif plagiarism_score > 30:
        recommendations.append({
            'type': 'info',
            'message': 'Plagiat modéré détecté',
            'action': 'Sensibiliser sur les bonnes pratiques de citation'
        })
    
    if ai_score > 80:
        recommendations.append({
            'type': 'warning',
            'message': 'Forte probabilité de génération par IA',
            'action': 'Demander une réécriture ou un entretien oral'
        })
    elif ai_score > 60:
        recommendations.append({
            'type': 'caution',
            'message': 'Probabilité modérée de génération par IA',
            'action': 'Poser des questions spécifiques sur le contenu'
        })
    
    if plagiarism_score < 20 and ai_score < 30:
        recommendations.append({
            'type': 'success',
            'message': 'Travail original détecté',
            'action': 'Aucune action nécessaire'
        })
    
    return recommendations

# Création des tables
@app.before_first_request
def create_tables():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)