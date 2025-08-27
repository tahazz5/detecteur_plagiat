# plagiarism_detector.py - Module de détection de plagiat
import re
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from typing import List, Dict, Tuple
import hashlib
import sqlite3
from datetime import datetime

class PlagiarismDetector:
    def __init__(self):
        """Initialiser le détecteur de plagiat"""
        self.setup_nltk()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.8,
            min_df=0.1
        )
        self.known_sources_db = 'known_sources.db'
        self.setup_database()
        self.common_phrases = self.load_common_phrases()
        
    def setup_nltk(self):
        """Télécharger les ressources NLTK nécessaires"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words(['english', 'french']))
        except:
            self.stop_words = set()
    
    def setup_database(self):
        """Initialiser la base de données des sources connues"""
        conn = sqlite3.connect(self.known_sources_db)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                content TEXT,
                content_hash TEXT,
                last_updated TIMESTAMP,
                source_type TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def load_common_phrases(self) -> List[str]:
        """Charger les phrases communément plagiées"""
        return [
            "in the modern world",
            "since the beginning of time",
            "throughout history",
            "in today's society",
            "it is widely accepted that",
            "research has shown that",
            "studies have proven that",
            "experts agree that",
            "it is important to note that",
            "in conclusion",
            "to summarize",
            "dans le contexte actuel",
            "de nos jours",
            "il est important de noter",
            "selon les experts",
            "les recherches montrent que",
            "en conclusion"
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Préprocesser le texte pour l'analyse"""
        # Nettoyer le texte
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Tokenisation et suppression des mots vides
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculer la similarité entre deux textes"""
        try:
            # Préprocesser les textes
            processed_text1 = self.preprocess_text(text1)
            processed_text2 = self.preprocess_text(text2)
            
            # TF-IDF similarity
            tfidf_matrix = self.vectorizer.fit_transform([processed_text1, processed_text2])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Sequence similarity (pour les correspondances exactes)
            sequence_similarity = SequenceMatcher(None, processed_text1, processed_text2).ratio()
            
            # N-gram similarity
            ngram_similarity = self.calculate_ngram_similarity(processed_text1, processed_text2)
            
            # Score composite
            final_score = (tfidf_similarity * 0.4 + sequence_similarity * 0.4 + ngram_similarity * 0.2)
            
            return min(final_score * 100, 100.0)  # Convertir en pourcentage
            
        except Exception as e:
            print(f"Erreur de calcul de similarité: {e}")
            return 0.0
    
    def calculate_ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Calculer la similarité basée sur les n-grammes"""
        def get_ngrams(text: str, n: int) -> set:
            words = text.split()
            return set([' '.join(words[i:i+n]) for i in range(len(words)-n+1)])
        
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def detect_common_phrases(self, text: str) -> List[Dict]:
        """Détecter les phrases communes suspectes"""
        detected_phrases = []
        text_lower = text.lower()
        
        for phrase in self.common_phrases:
            if phrase.lower() in text_lower:
                # Trouver la position exacte
                start_pos = text_lower.find(phrase.lower())
                detected_phrases.append({
                    'phrase': phrase,
                    'position': start_pos,
                    'context': text[max(0, start_pos-50):start_pos+len(phrase)+50]
                })
        
        return detected_phrases
    
    def search_web_sources(self, text: str, max_sources: int = 5) -> List[Dict]:
        """Rechercher des sources potentielles sur le web (simulation)"""
        # Dans une vraie implémentation, vous utiliseriez des APIs comme:
        # - Google Custom Search API
        # - Bing Search API
        # - Academic databases APIs
        
        potential_sources = []
        
        # Simulation de sources académiques
        academic_sources = [
            {
                'url': 'https://scholar.google.com/example1',
                'title': 'Academic Paper on AI',
                'type': 'academic',
                'content': 'Artificial intelligence has revolutionized modern computing...'
            },
            {
                'url': 'https://en.wikipedia.org/wiki/Machine_Learning',
                'title': 'Machine Learning - Wikipedia',
                'type': 'encyclopedia',
                'content': 'Machine learning is a subset of artificial intelligence...'
            },
            {
                'url': 'https://www.nature.com/articles/nature12345',
                'title': 'Nature Article on Deep Learning',
                'type': 'journal',
                'content': 'Deep learning algorithms have shown remarkable progress...'
            }
        ]
        
        # Calculer la similarité avec chaque source
        for source in academic_sources[:max_sources]:
            similarity = self.calculate_similarity(text, source['content'])
            if similarity > 15:  # Seuil minimum
                potential_sources.append({
                    'url': source['url'],
                    'title': source['title'],
                    'type': source['type'],
                    'similarity': round(similarity, 2),
                    'matched_segments': self.find_matching_segments(text, source['content'])
                })
        
        return sorted(potential_sources, key=lambda x: x['similarity'], reverse=True)
    
    def find_matching_segments(self, text1: str, text2: str, min_length: int = 50) -> List[Dict]:
        """Trouver les segments de texte qui correspondent"""
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        
        matching_segments = []
        
        for i, sent1 in enumerate(sentences1):
            if len(sent1) < min_length:
                continue
                
            for j, sent2 in enumerate(sentences2):
                similarity = SequenceMatcher(None, sent1, sent2).ratio()
                if similarity > 0.6:  # 60% de similarité
                    matching_segments.append({
                        'original_sentence': sent1,
                        'source_sentence': sent2,
                        'similarity': round(similarity * 100, 2),
                        'position_in_text': i,
                        'position_in_source': j
                    })
        
        return matching_segments
    
    def check_submission_database(self, text: str) -> List[Dict]:
        """Vérifier contre les soumissions précédentes"""
        # Simulation - dans une vraie implémentation, 
        # vous compareriez avec toutes les soumissions précédentes
        return []
    
    def detect_plagiarism(self, text: str) -> Dict:
        """Fonction principale de détection de plagiat"""
        if len(text.strip()) < 50:
            return {
                'error': 'Texte trop court pour l\'analyse',
                'score': 0,
                'sources': [],
                'common_phrases': [],
                'recommendations': []
            }
        
        print(f"Analyse du plagiat pour un texte de {len(text)} caractères...")
        
        # Détection des phrases communes
        common_phrases = self.detect_common_phrases(text)
        
        # Recherche de sources web
        web_sources = self.search_web_sources(text)
        
        # Vérification dans la base de soumissions
        submission_matches = self.check_submission_database(text)
        
        # Calcul du score global de plagiat
        plagiarism_score = self.calculate_overall_score(
            common_phrases, web_sources, submission_matches, text
        )
        
        # Analyse des statistiques textuelles
        text_stats = self.analyze_text_statistics(text)
        
        result = {
            'score': round(plagiarism_score, 2),
            'sources': web_sources + submission_matches,
            'common_phrases': common_phrases,
            'text_statistics': text_stats,
            'analysis_details': {
                'total_sources_checked': len(web_sources) + len(submission_matches),
                'high_similarity_sources': len([s for s in web_sources if s['similarity'] > 50]),
                'common_phrases_found': len(common_phrases)
            },
            'recommendations': self.generate_plagiarism_recommendations(plagiarism_score)
        }
        
        return result
    
    def calculate_overall_score(self, common_phrases: List, web_sources: List, 
                               submission_matches: List, text: str) -> float:
        """Calculer le score global de plagiat"""
        score = 0.0
        
        # Score basé sur les phrases communes (max 30 points)
        if common_phrases:
            common_phrase_score = min(len(common_phrases) * 8, 30)
            score += common_phrase_score
        
        # Score basé sur les sources web (max 50 points)
        if web_sources:
            max_similarity = max([s['similarity'] for s in web_sources])
            web_score = min(max_similarity * 0.8, 50)
            score += web_score
        
        # Score basé sur les soumissions précédentes (max 40 points)
        if submission_matches:
            max_submission_similarity = max([s['similarity'] for s in submission_matches])
            submission_score = min(max_submission_similarity * 0.9, 40)
            score += submission_score
        
        # Ajustements basés sur la longueur et la complexité
        word_count = len(text.split())
        if word_count < 100:
            score += 10  # Pénalité pour texte très court
        
        # Détection de patterns suspects
        if self.has_suspicious_patterns(text):
            score += 15
        
        return min(score, 100.0)
    
    def has_suspicious_patterns(self, text: str) -> bool:
        """Détecter des patterns suspects dans le texte"""
        # Citations multiples sans source
        citation_pattern = r'"[^"]{20,}"'
        citations = re.findall(citation_pattern, text)
        
        # Changements soudains de style
        sentences = sent_tokenize(text)
        if len(sentences) > 5:
            # Analyser la variation de longueur des phrases
            sentence_lengths = [len(sent.split()) for sent in sentences]
            avg_length = np.mean(sentence_lengths)
            std_length = np.std(sentence_lengths)
            
            # Si la variation est trop grande, c'est suspect
            if std_length > avg_length * 0.8:
                return True
        
        # Présence de références incomplètes
        incomplete_refs = re.findall(r'\(.*\d{4}.*\)', text)
        
        return len(citations) > 3 or len(incomplete_refs) > 2
    
    def analyze_text_statistics(self, text: str) -> Dict:
        """Analyser les statistiques du texte"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': round(len(words) / len(sentences), 2) if sentences else 0,
            'unique_words': len(set(word.lower() for word in words if word.isalpha())),
            'readability_score': self.calculate_readability(text),
            'complexity_indicators': self.analyze_complexity(text)
        }
    
    def calculate_readability(self, text: str) -> float:
        """Calculer un score de lisibilité simple"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Score simple basé sur la longueur moyenne des phrases
        if avg_sentence_length < 10:
            return 85.0  # Facile
        elif avg_sentence_length < 20:
            return 65.0  # Moyen
        else:
            return 45.0  # Difficile
    
    def analyze_complexity(self, text: str) -> Dict:
        """Analyser la complexité du texte"""
        words = word_tokenize(text.lower())
        
        # Compter les mots complexes (plus de 3 syllabes - approximation)
        complex_words = [w for w in words if len(w) > 8 and w.isalpha()]
        
        return {
            'complex_word_ratio': round(len(complex_words) / len(words), 3) if words else 0,
            'average_word_length': round(np.mean([len(w) for w in words if w.isalpha()]), 2) if words else 0,
            'vocabulary_diversity': round(len(set(words)) / len(words), 3) if words else 0
        }
    
    def generate_plagiarism_recommendations(self, score: float) -> List[str]:
        """Générer des recommandations basées sur le score"""
        recommendations = []
        
        if score > 70:
            recommendations.extend([
                "Investigation approfondie requise",
                "Convoquer l'étudiant pour discussion",
                "Vérifier toutes les sources identifiées",
                "Considérer une réécriture complète"
            ])
        elif score > 50:
            recommendations.extend([
                "Examen manuel des sections suspectes",
                "Demander les sources utilisées",
                "Sensibilisation aux règles de citation"
            ])
        elif score > 30:
            recommendations.extend([
                "Vérification ponctuelle recommandée",
                "Formation sur les bonnes pratiques de citation",
                "Surveillance renforcée pour les prochains travaux"
            ])
        elif score > 15:
            recommendations.extend([
                "Plagiat mineur détecté",
                "Rappel des règles de citation",
                "Suivi préventif"
            ])
        else:
            recommendations.append("Travail original - Aucune action nécessaire")
        
        return recommendations