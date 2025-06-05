from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

from app import db
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(60), nullable=False, unique=True)
    password_hash = db.Column(db.String(128))
    time_zone = db.Column(db.String(50), nullable=False, default='UTC')
    is_admin = db.Column(db.Boolean, default=False)
    credits = db.Column(db.Integer, default=10)  # Start with 10 free credits

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def has_sufficient_credits(self, required_credits):
        return self.credits >= required_credits
    
    def deduct_credits(self, amount):
        if self.has_sufficient_credits(amount):
            self.credits -= amount
            return True
        return False

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    credits_used = db.Column(db.Integer, nullable=False, default=1)
    
    user = db.relationship('User', backref=db.backref('questions', lazy=True))
    responses = db.relationship('Response', backref='question', lazy=True, cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Question {self.id}: {self.content[:50]}...>'

class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.id'), nullable=False)
    llm_name = db.Column(db.String(50), nullable=False)  # e.g., 'gpt-4', 'claude', etc.
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    tokens_used = db.Column(db.Integer, nullable=True)
    
    def __repr__(self):
        return f'<Response from {self.llm_name} for Question {self.question_id}>'

class LogEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    actor_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    category = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)

    actor = db.relationship('User', backref=db.backref('log_entries', lazy=True))

    def __repr__(self):
        return f'<LogEntry {self.timestamp} - {self.category}>'
