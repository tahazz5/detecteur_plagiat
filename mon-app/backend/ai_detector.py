
# ai_detector.py - Module de détection de texte généré par IA
import re
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import pickle
import os
from typing import Dict, List, Tuple
import requests
import json

class AIDetector:
    def __init__(self):
        """Initialiser le détecteur d'IA"""
        self.setup_nltk()
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        self.load_or_train_model()
        self.ai_patterns = self.load_ai_patterns()
        
    def setup_nltk(self):
        """Configuration NLTK"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
    
    def load_ai_patterns(self) -> Dict:
        """Charger les patterns caractéristiques des textes IA"""
        return {
            'transition_words': [
                'furthermore', 'moreover', 'additionally', 'consequently',
                'therefore', 'thus', 'hence', 'nevertheless', 'however',
                'de plus', 'par ailleurs', 'en outre', 'cependant', 
                'néanmoins', 'par conséquent', 'ainsi'
            ],
            'hedge_words': [
                'might', 'could', 'would', 'should', 'may', 'perhaps',
                'possibly', 'likely', 'potentially', 'arguably',
                'pourrait', 'devrait', 'peut-être', 'probablement',
                'possiblement', 'vraisemblablement'
            ],
            'formal_structures': [
                r'it is important to note that',
                r'it should be noted that',
                r'it is worth mentioning that',
                r'research suggests that',
                r'studies indicate that',
                r'il est important de noter que',
                r'il convient de mentionner que',
                r'les recherches suggèrent que',
                r'les études indiquent que'
            ],
            'ai_phrases': [
                'as an ai language model',
                'i don\'t have personal opinions',
                'in my analysis',
                'based on my understanding',
                'from my perspective',
                'en tant que modèle de langage',
                'selon mon analyse',
                'selon ma compréhension'
            ]
        }
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculer la perplexité du texte (approximation)"""
        words = word_tokenize(text.lower())
        if len(words) < 10:
            return 100.0
        
        # Calcul simplifié de perplexité basé sur la fréquence des mots
        word_freq = Counter(words)
        total_words = len(words)
        
        # Calcul de la probabilité moyenne
        prob_sum = 0
        for word in words:
            prob = word_freq[word] / total_words
            if prob > 0:
                prob_sum += math.log(prob)
        
        avg_log_prob = prob_sum / len(words)
        perplexity = math.exp(-avg_log_prob)
        
        return min(perplexity, 1000.0)  # Cap à 1000
    
    def analyze_sentence_structure(self, text: str) -> Dict:
        """Analyser la structure des phrases"""
        sentences = sent_tokenize(text)
        
        if not sentences:
            return {'uniformity': 0, 'complexity': 0, 'variation': 0}
        
        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        
        # Uniformité des longueurs de phrases
        if len(sentence_lengths) > 1:
            uniformity = 1.0 - (np.std(sentence_lengths) / np.mean(sentence_lengths))
        else:
            uniformity = 1.0
        
        # Complexité moyenne
        avg_length = np.mean(sentence_lengths)
        complexity = min(avg_length / 20.0, 1.0)  # Normaliser sur 20 mots
        
        # Variation structurelle
        variation = len(set(sentence_lengths)) / len(sentence_lengths)
        
        return {
            'uniformity': round(uniformity, 3),
            'complexity': round(complexity, 3),
            'variation': round(variation, 3),
            'avg_sentence_length': round(avg_length, 2)
        }
    
    def detect_ai_patterns(self, text: str) -> Dict:
        """Détecter les patterns spécifiques à l'IA"""
        text_lower = text.lower()
        
        pattern_scores = {
            'transition_words': 0,
            'hedge_words': 0,
            'formal_structures': 0,
            'ai_phrases': 0,
            'repetitive_patterns': 0
        }
        
        # Compter les mots de transition
        transition_count = sum(1 for word in self.ai_patterns['transition_words'] 
                              if word in text_lower)
        pattern_scores['transition_words'] = min(transition_count * 10, 100)
        
        # Compter les mots d'hésitation/hedge words
        hedge_count = sum(1 for word in self.ai_patterns['hedge_words'] 
                         if word in text_lower)
        pattern_scores['hedge_words'] = min(hedge_count * 8, 100)
        
        # Détecter les structures formelles
        formal_count = sum(1 for pattern in self.ai_patterns['formal_structures'] 
                          if re.search(pattern, text_lower))
        pattern_scores['formal_structures'] = min(formal_count * 15, 100)
        
        # Détecter les phrases typiques d'IA
        ai_phrase_count = sum(1 for phrase in self.ai_patterns['ai_phrases'] 
                             if phrase in text_lower)
        pattern_scores['ai_phrases'] = min(ai_phrase_count * 25, 100)
        
        # Détecter les patterns répétitifs
        pattern_scores['repetitive_patterns'] = self.detect_repetitive_patterns(text)
        
        return pattern_scores
    
    def detect_repetitive_patterns(self, text: str) -> float:
        """Détecter les patterns répétitifs dans le texte"""
        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            return 0.0
        
        # Analyser les débuts de phrases
        sentence_starts = [sent.strip()[:20].lower() for sent in sentences if sent.strip()]
        start_similarity = len(sentence_starts) - len(set(sentence_starts))
        
        # Analyser les structures grammaticales répétitives
        structure_patterns = []
        for sent in sentences:
            words = word_tokenize(sent)[:5]  # 5 premiers mots
            if len(words) >= 2:
                pattern = ' '.join([w if w.isalpha() else 'X' for w in words])
                structure_patterns.append(pattern)
        
        structure_repetition = len(structure_patterns) - len(set(structure_patterns))
        
        # Score composite
        total_repetition = start_similarity + structure_repetition
        return min(total_repetition * 15, 100)
    
    def analyze_vocabulary_diversity(self, text: str) -> Dict:
        """Analyser la diversité du vocabulaire"""
        words = [w.lower() for w in word_tokenize(text) if w.isalpha() and len(w) > 2]
        
        if len(words) < 10:
            return {'diversity': 0, 'uniqueness': 0, 'sophistication': 0}
        
        unique_words = set(words)
        
        # Diversité lexicale (Type-Token Ratio)
        diversity = len(unique_words) / len(words)
        
        # Unicité (mots utilisés une seule fois)
        word_counts = Counter(words)
        unique_once = sum(1 for count in word_counts.values() if count == 1)
        uniqueness = unique_once / len(unique_words)
        
        # Sophistication (mots longs et rares)
        sophisticated_words = [w for w in unique_words if len(w) > 7]
        sophistication = len(sophisticated_words) / len(unique_words)
        
        return {
            'diversity': round(diversity, 3),
            'uniqueness': round(uniqueness, 3),
            'sophistication': round(sophistication, 3)
        }
    
    def extract_features(self, text: str) -> np.ndarray:
        """Extraire les caractéristiques pour le modèle ML"""
        # Caractéristiques de base
        word_count = len(word_tokenize(text))
        sent_count = len(sent_tokenize(text))
        avg_word_length = np.mean([len(w) for w in word_tokenize(text) if w.isalpha()])
        
        # Perplexité
        perplexity = self.calculate_perplexity(text)
        
        # Structure des phrases
        structure = self.analyze_sentence_structure(text)
        
        # Patterns IA
        ai_patterns = self.detect_ai_patterns(text)
        
        # Diversité vocabulaire
        vocab = self.analyze_vocabulary_diversity(text)
        
        # Caractéristiques stylistiques
        punct_density = len(re.findall(r'[.,;:!?]', text)) / len(text)
        exclamation_ratio = text.count('!') / max(sent_count, 1)
        question_ratio = text.count('?') / max(sent_count, 1)
        
        # Combiner toutes les caractéristiques
        features = np.array([
            word_count / 1000,  # Normaliser
            avg_word_length,
            structure['uniformity'],
            structure['complexity'],
            structure['variation'],
            perplexity / 100,  # Normaliser
            ai_patterns['transition_words'] / 100,
            ai_patterns['hedge_words'] / 100,
            ai_patterns['formal_structures'] / 100,
            ai_patterns['ai_phrases'] / 100,
            ai_patterns['repetitive_patterns'] / 100,
            vocab['diversity'],
            vocab['uniqueness'],
            vocab['sophistication'],
            punct_density,
            exclamation_ratio,
            question_ratio
        ])
        
        return features
    
    def load_or_train_model(self):
        """Charger ou entraîner le modèle de classification"""
        model_path = 'ai_detector_model.pkl'
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("Modèle de détection IA chargé avec succès")
                return
            except:
                print("Erreur lors du chargement du modèle, réentraînement...")
        
        # Entraîner un nouveau modèle avec des données simulées
        self.train_model()
        
        # Sauvegarder le modèle
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
        except:
            print("Impossible de sauvegarder le modèle")
    
    def train_model(self):
        """Entraîner le modèle avec des données simulées"""
        print("Entraînement du modèle de détection IA...")
        
        # Données simulées - Dans une vraie implémentation,
        # vous auriez des milliers d'exemples étiquetés
        human_texts = [
            "Je pense que ce projet est vraiment intéressant. Ça me rappelle mes études, honnêtement ! J'ai eu quelques difficultés au début mais maintenant ça va mieux. Mon prof nous a expliqué les bases mais c'était un peu compliqué à comprendre au début.",
            "Alors voilà ce que j'en pense... C'est pas évident de tout comprendre d'un coup. Des fois je me dis que c'est trop dur, puis après ça va. J'ai essayé plusieurs méthodes et celle-ci marche plutôt bien pour moi.",
            "J'avoue que j'étais perdu au début ! Mais bon, avec de la pratique on s'améliore. J'ai même réussi à expliquer ça à un ami hier. C'est fou comme on progresse sans s'en rendre compte."
        ]
        
        ai_texts = [
            "It is important to note that this topic represents a significant advancement in the field. Furthermore, research indicates that the methodology employed demonstrates considerable promise. However, it should be acknowledged that certain limitations exist. Nevertheless, the overall findings suggest positive outcomes.",
            "The analysis reveals several key insights. Firstly, the data demonstrates clear patterns. Additionally, the statistical significance of these results cannot be overlooked. Moreover, the implications for future research are substantial. Consequently, further investigation is warranted.",
            "In examining this subject matter, it becomes evident that multiple factors contribute to the observed phenomena. Therefore, a comprehensive approach is necessary. Furthermore, the interdisciplinary nature of this field requires careful consideration of various perspectives."
        ]
        
        # Préparer les données d'entraînement
        texts = human_texts + ai_texts
        labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0 = humain, 1 = IA
        
        # Extraire les caractéristiques
        features = np.array([self.extract_features(text) for text in texts])
        
        # Entraîner le modèle
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(features, labels)
        
        print("Modèle entraîné avec succès")
    
    def detect_ai_text(self, text: str) -> Dict:
        """Fonction principale de détection de texte IA"""
        if len(text.strip()) < 50:
            return {
                'error': 'Texte trop court pour l\'analyse IA',
                'score': 0,
                'confidence': 0,
                'features': {},
                'recommendations': []
            }
        
        print(f"Analyse IA pour un texte de {len(text)} caractères...")
        
        try:
            # Extraire les caractéristiques
            features = self.extract_features(text)
            
            # Prédiction avec le modèle ML
            if self.model is not None:
                ai_probability = self.model.predict_proba([features])[0][1]
                ml_score = ai_probability * 100
            else:
                ml_score = 50.0  # Score neutre si pas de modèle
            
            # Analyse des patterns
            patterns = self.detect_ai_patterns(text)
            pattern_score = np.mean(list(patterns.values()))
            
            # Analyse de la perplexité
            perplexity = self.calculate_perplexity(text)
            perplexity_score = max(0, 100 - perplexity * 2)  # Plus la perplexité est faible, plus c'est suspect
            
            # Score composite
            final_score = (ml_score * 0.4 + pattern_score * 0.4 + perplexity_score * 0.2)
            
            # Analyse détaillée
            structure = self.analyze_sentence_structure(text)
            vocab = self.analyze_vocabulary_diversity(text)
            
            # Détection de caractéristiques spécifiques
            suspicious_features = self.identify_suspicious_features(text, patterns, structure, vocab)
            
            result = {
                'score': round(final_score, 2),
                'confidence': round(abs(final_score - 50) / 50 * 100, 2),
                'ml_prediction': round(ml_score, 2),
                'pattern_analysis': patterns,
                'perplexity': round(perplexity, 2),
                'text_structure': structure,
                'vocabulary_analysis': vocab,
                'suspicious_features': suspicious_features,
                'detailed_analysis': {
                    'coherence_score': self.calculate_coherence_score(text),
                    'formality_score': self.calculate_formality_score(text),
                    'creativity_score': self.calculate_creativity_score(text),
                    'human_error_indicators': self.detect_human_errors(text)
                },
                'recommendations': self.generate_ai_recommendations(final_score, suspicious_features)
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Erreur lors de l\'analyse IA: {str(e)}',
                'score': 50.0,
                'confidence': 0,
                'features': {},
                'recommendations': ['Erreur d\'analyse - Vérification manuelle recommandée']
            }
    
    def identify_suspicious_features(self, text: str, patterns: Dict, structure: Dict, vocab: Dict) -> List[str]:
        """Identifier les caractéristiques suspectes du texte"""
        suspicious = []
        
        # Structure trop uniforme
        if structure['uniformity'] > 0.8:
            suspicious.append("Structure de phrases trop uniforme")
        
        # Trop de mots de transition
        if patterns['transition_words'] > 50:
            suspicious.append("Usage excessif de mots de transition")
        
        # Patterns répétitifs
        if patterns['repetitive_patterns'] > 40:
            suspicious.append("Structures répétitives détectées")
        
        # Vocabulaire trop sophistiqué
        if vocab['sophistication'] > 0.3:
            suspicious.append("Vocabulaire exceptionnellement sophistiqué")
        
        # Diversité lexicale suspecte
        if vocab['diversity'] < 0.3:
            suspicious.append("Diversité lexicale limitée")
        
        # Présence de phrases typiques d'IA
        if patterns['ai_phrases'] > 0:
            suspicious.append("Phrases caractéristiques d'IA détectées")
        
        return suspicious
    
    def calculate_coherence_score(self, text: str) -> float:
        """Calculer un score de cohérence du texte"""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 50.0
        
        # Mesure simplifiée de cohérence basée sur la continuité thématique
        word_overlap = 0
        total_comparisons = 0
        
        for i in range(len(sentences) - 1):
            words1 = set(word_tokenize(sentences[i].lower()))
            words2 = set(word_tokenize(sentences[i + 1].lower()))
            
            if words1 and words2:
                overlap = len(words1.intersection(words2))
                word_overlap += overlap / max(len(words1), len(words2))
                total_comparisons += 1
        
        coherence = (word_overlap / total_comparisons) * 100 if total_comparisons > 0 else 50
        return round(coherence, 2)
    
    def calculate_formality_score(self, text: str) -> float:
        """Calculer un score de formalité du texte"""
        formal_indicators = [
            'furthermore', 'moreover', 'consequently', 'nevertheless',
            'therefore', 'however', 'additionally', 'specifically'
        ]
        
        informal_indicators = [
            'gonna', 'wanna', 'kinda', 'really', 'pretty', 'very',
            'just', 'like', 'you know', 'I mean'
        ]
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_indicators if word in text_lower)
        informal_count = sum(1 for word in informal_indicators if word in text_lower)
        
        if formal_count + informal_count == 0:
            return 50.0
        
        formality = (formal_count / (formal_count + informal_count)) * 100
        return round(formality, 2)
    
    def calculate_creativity_score(self, text: str) -> float:
        """Calculer un score de créativité du texte"""
        # Mesures de créativité simplifiées
        sentences = sent_tokenize(text)
        
        # Variété des longueurs de phrases
        if len(sentences) > 1:
            lengths = [len(word_tokenize(s)) for s in sentences]
            length_variety = np.std(lengths) / np.mean(lengths) if np.mean(lengths) > 0 else 0
        else:
            length_variety = 0
        
        # Utilisation de métaphores/comparaisons
        metaphor_indicators = ['like', 'as', 'compared to', 'similar to', 'resembles']
        metaphor_count = sum(1 for indicator in metaphor_indicators 
                           if indicator in text.lower())
        
        # Questions rhétoriques
        question_count = text.count('?')
        
        # Score composite
        creativity = min((length_variety * 30 + metaphor_count * 15 + question_count * 10), 100)
        return round(creativity, 2)
    
    def detect_human_errors(self, text: str) -> Dict:
        """Détecter les erreurs typiquement humaines"""
        errors = {
            'typos': 0,
            'grammar_inconsistencies': 0,
            'informal_language': 0,
            'redundancy': 0
        }
        
        # Détection de fautes de frappe courantes (simplifiée)
        common_typos = ['teh', 'adn', 'recieve', 'seperate', 'definately']
        errors['typos'] = sum(1 for typo in common_typos if typo in text.lower())
        
        # Détection de langage informel
        informal_words = ['yeah', 'ok', 'gonna', 'wanna', 'kinda']
        errors['informal_language'] = sum(1 for word in informal_words 
                                        if word in text.lower())
        
        # Détection de redondances
        words = word_tokenize(text.lower())
        word_counts = Counter(words)
        errors['redundancy'] = sum(1 for count in word_counts.values() if count > 5)
        
        return errors
    
    def generate_ai_recommendations(self, score: float, suspicious_features: List[str]) -> List[str]:
        """Générer des recommandations basées sur l'analyse IA"""
        recommendations = []
        
        if score > 80:
            recommendations.extend([
                "Forte probabilité de génération par IA",
                "Vérification approfondie recommandée",
                "Demander une réécriture ou un entretien oral",
                "Examiner les sources et le processus de rédaction"
            ])
        elif score > 60:
            recommendations.extend([
                "Probabilité modérée de génération par IA",
                "Poser des questions spécifiques sur le contenu",
                "Vérifier la compréhension des concepts clés"
            ])
        elif score > 40:
            recommendations.extend([
                "Quelques indicateurs d'IA détectés",
                "Surveillance recommandée",
                "Sensibiliser aux politiques d'utilisation d'IA"
            ])
        else:
            recommendations.append("Faible probabilité d'IA - Travail probablement humain")
        
        # Recommandations spécifiques aux caractéristiques suspectes
        if suspicious_features:
            recommendations.append("Caractéristiques suspectes détectées:")
            recommendations.extend([f"• {feature}" for feature in suspicious_features])
        
        return recommendations