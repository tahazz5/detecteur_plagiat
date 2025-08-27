# models.py - Modèles de base de données étendus
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid
import json

db = SQLAlchemy()

class Student(db.Model):
    """Modèle pour les étudiants"""
    __tablename__ = 'students'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_number = db.Column(db.String(50), unique=True, nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    class_group = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relations
    submissions = db.relationship('Submission', backref='student', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_number': self.student_number,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'class_group': self.class_group,
            'created_at': self.created_at.isoformat()
        }

class Assignment(db.Model):
    """Modèle pour les devoirs/assignations"""
    __tablename__ = 'assignments'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    course_code = db.Column(db.String(50))
    instructor_name = db.Column(db.String(100))
    due_date = db.Column(db.DateTime)
    max_plagiarism_threshold = db.Column(db.Float, default=30.0)
    max_ai_threshold = db.Column(db.Float, default=40.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relations
    submissions = db.relationship('Submission', backref='assignment', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'course_code': self.course_code,
            'instructor_name': self.instructor_name,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'max_plagiarism_threshold': self.max_plagiarism_threshold,
            'max_ai_threshold': self.max_ai_threshold,
            'created_at': self.created_at.isoformat()
        }

class Submission(db.Model):
    """Modèle pour les soumissions"""
    __tablename__ = 'submissions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    student_id = db.Column(db.String(36), db.ForeignKey('students.id'))
    assignment_id = db.Column(db.String(36), db.ForeignKey('assignments.id'))
    filename = db.Column(db.String(255))
    original_filename = db.Column(db.String(255))
    file_size = db.Column(db.Integer)
    file_type = db.Column(db.String(10))
    text_content = db.Column(db.Text)
    text_hash = db.Column(db.String(64))  # SHA-256 du contenu
    
    # Scores d'analyse
    plagiarism_score = db.Column(db.Float)
    ai_score = db.Column(db.Float)
    overall_risk_score = db.Column(db.Float)
    
    # Résultats détaillés (JSON)
    plagiarism_analysis = db.Column(db.Text)  # JSON
    ai_analysis = db.Column(db.Text)  # JSON
    text_statistics = db.Column(db.Text)  # JSON
    
    # Statut et métadonnées
    status = db.Column(db.String(20), default='pending')  # pending, analyzed, reviewed, flagged
    needs_review = db.Column(db.Boolean, default=False)
    reviewed_by = db.Column(db.String(100))
    review_notes = db.Column(db.Text)
    review_date = db.Column(db.DateTime)
    
    # Timestamps
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyzed_at = db.Column(db.DateTime)
    
    # Relations
    detected_sources = db.relationship('DetectedSource', backref='submission', lazy=True, cascade='all, delete-orphan')
    similarity_matches = db.relationship('SimilarityMatch', backref='submission', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self, include_content=False):
        result = {
            'id': self.id,
            'student_id': self.student_id,
            'assignment_id': self.assignment_id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_size': self.file_size,
            'file_type': self.file_type,
            'plagiarism_score': self.plagiarism_score,
            'ai_score': self.ai_score,
            'overall_risk_score': self.overall_risk_score,
            'status': self.status,
            'needs_review': self.needs_review,
            'reviewed_by': self.reviewed_by,
            'review_notes': self.review_notes,
            'review_date': self.review_date.isoformat() if self.review_date else None,
            'submitted_at': self.submitted_at.isoformat(),
            'analyzed_at': self.analyzed_at.isoformat() if self.analyzed_at else None
        }
        
        if include_content:
            result.update({
                'text_content': self.text_content,
                'plagiarism_analysis': json.loads(self.plagiarism_analysis) if self.plagiarism_analysis else None,
                'ai_analysis': json.loads(self.ai_analysis) if self.ai_analysis else None,
                'text_statistics': json.loads(self.text_statistics) if self.text_statistics else None
            })
        
        return result

class DetectedSource(db.Model):
    """Sources potentielles de plagiat détectées"""
    __tablename__ = 'detected_sources'
    
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.String(36), db.ForeignKey('submissions.id'))
    source_url = db.Column(db.String(1000))
    source_title = db.Column(db.String(500))
    source_type = db.Column(db.String(50))  # web, academic, book, etc.
    similarity_score = db.Column(db.Float)
    matched_text_segments = db.Column(db.Text)  # JSON array
    confidence_level = db.Column(db.String(20))  # high, medium, low
    
    def to_dict(self):
        return {
            'id': self.id,
            'source_url': self.source_url,
            'source_title': self.source_title,
            'source_type': self.source_type,
            'similarity_score': self.similarity_score,
            'matched_text_segments': json.loads(self.matched_text_segments) if self.matched_text_segments else [],
            'confidence_level': self.confidence_level
        }

class SimilarityMatch(db.Model):
    """Correspondances avec d'autres soumissions"""
    __tablename__ = 'similarity_matches'
    
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.String(36), db.ForeignKey('submissions.id'))
    matched_submission_id = db.Column(db.String(36), db.ForeignKey('submissions.id'))
    similarity_score = db.Column(db.Float)
    matched_segments = db.Column(db.Text)  # JSON
    match_type = db.Column(db.String(20))  # exact, paraphrase, structural
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relation vers la soumission correspondante
    matched_submission = db.relationship('Submission', foreign_keys=[matched_submission_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'matched_submission_id': self.matched_submission_id,
            'similarity_score': self.similarity_score,
            'matched_segments': json.loads(self.matched_segments) if self.matched_segments else [],
            'match_type': self.match_type,
            'created_at': self.created_at.isoformat()
        }

class AnalysisLog(db.Model):
    """Log des analyses effectuées"""
    __tablename__ = 'analysis_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.String(36), db.ForeignKey('submissions.id'))
    analysis_type = db.Column(db.String(20))  # plagiarism, ai, full
    processing_time = db.Column(db.Float)  # en secondes
    success = db.Column(db.Boolean)
    error_message = db.Column(db.Text)
    algorithm_version = db.Column(db.String(20))
    parameters_used = db.Column(db.Text)  # JSON
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'submission_id': self.submission_id,
            'analysis_type': self.analysis_type,
            'processing_time': self.processing_time,
            'success': self.success,
            'error_message': self.error_message,
            'algorithm_version': self.algorithm_version,
            'parameters_used': json.loads(self.parameters_used) if self.parameters_used else {},
            'timestamp': self.timestamp.isoformat()
        }

class SystemSettings(db.Model):
    """Paramètres système"""
    __tablename__ = 'system_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.Text)
    description = db.Column(db.Text)
    data_type = db.Column(db.String(20))  # string, int, float, bool, json
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def get_value(self):
        """Retourner la valeur avec le bon type"""
        if self.data_type == 'int':
            return int(self.value)
        elif self.data_type == 'float':
            return float(self.value)
        elif self.data_type == 'bool':
            return self.value.lower() in ('true', '1', 'yes')
        elif self.data_type == 'json':
            return json.loads(self.value)
        else:
            return self.value
    
    def set_value(self, value):
        """Définir la valeur avec conversion automatique"""
        if self.data_type == 'json':
            self.value = json.dumps(value)
        else:
            self.value = str(value)
    
    def to_dict(self):
        return {
            'key': self.key,
            'value': self.get_value(),
            'description': self.description,
            'data_type': self.data_type,
            'updated_at': self.updated_at.isoformat()
        }