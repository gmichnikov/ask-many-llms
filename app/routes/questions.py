from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_required, current_user
from app import db
from app.models import Question, Response
from app.forms import QuestionForm
from app.services.llm_service import LLMService

bp = Blueprint('questions', __name__)
llm_service = LLMService()

@bp.route('/ask', methods=['GET', 'POST'])
@login_required
def ask_question():
    form = QuestionForm()
    if form.validate_on_submit():
        # Check if user has enough credits
        if not current_user.has_sufficient_credits(1):
            flash('You do not have enough credits to ask a question.', 'error')
            return redirect(url_for('questions.ask_question'))
        
        # Create the question
        question = Question(
            content=form.content.data,
            user_id=current_user.id,
            credits_used=1
        )
        db.session.add(question)
        
        # Deduct credits
        current_user.deduct_credits(1)
        
        try:
            # Get responses from LLMs
            responses = llm_service.get_responses(question.content)
            
            # Save responses
            for response_data in responses:
                response = Response(
                    question=question,
                    llm_name=response_data['llm_name'],
                    content=response_data['content'],
                    tokens_used=response_data['tokens_used']
                )
                db.session.add(response)
            
            db.session.commit()
            flash('Your question has been submitted and responses are ready!', 'success')
            return redirect(url_for('questions.view_question', question_id=question.id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('questions.ask_question'))
    
    return render_template('questions/ask.html', form=form)

@bp.route('/question/<int:question_id>')
@login_required
def view_question(question_id):
    question = Question.query.get_or_404(question_id)
    if question.user_id != current_user.id and not current_user.is_admin:
        flash('You do not have permission to view this question.', 'error')
        return redirect(url_for('main.index'))
    return render_template('questions/view.html', question=question)

@bp.route('/questions')
@login_required
def list_questions():
    questions = Question.query.filter_by(user_id=current_user.id).order_by(Question.timestamp.desc()).all()
    return render_template('questions/list.html', questions=questions) 