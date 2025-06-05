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
                    model_name=response_data['metadata'].get('model', response_data['llm_name']),
                    content=response_data['content'],
                    input_tokens=response_data['metadata']['input_tokens'],
                    output_tokens=response_data['metadata']['output_tokens'],
                    total_tokens=response_data['metadata']['total_tokens']
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
    
    # Get responses for this question
    responses = []
    for response in question.responses:
        # Calculate cost based on the model
        cost = 0.0
        if response.llm_name == 'OpenAI':
            cost = (response.input_tokens / 1_000_000 * 0.40) + (response.output_tokens / 1_000_000 * 1.60)
        elif response.llm_name == 'Claude':
            cost = (response.input_tokens / 1_000_000 * 0.80) + (response.output_tokens / 1_000_000 * 4.00)
        elif response.llm_name == 'Gemini':
            cost = (response.input_tokens / 1_000_000 * 0.10) + (response.output_tokens / 1_000_000 * 0.40)
        
        responses.append({
            'llm_name': response.llm_name,
            'content': response.content,
            'metadata': {
                'input_tokens': response.input_tokens,
                'output_tokens': response.output_tokens,
                'total_tokens': response.total_tokens,
                'cost': cost,
                'model': response.model_name,
                'finish_reason': None,
                'stop_reason': None,
                'safety_ratings': None
            }
        })
    
    return render_template('questions/view.html', question=question, responses=responses)

@bp.route('/questions')
@login_required
def list_questions():
    questions = Question.query.filter_by(user_id=current_user.id).order_by(Question.timestamp.desc()).all()
    return render_template('questions/list.html', questions=questions)

@bp.route('/admin/questions')
@login_required
def admin_list_questions():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('main.index'))
    
    questions = Question.query.order_by(Question.timestamp.desc()).all()
    return render_template('questions/admin_list.html', questions=questions)

@bp.route('/admin/question/<int:question_id>/delete', methods=['POST'])
@login_required
def admin_delete_question(question_id):
    if not current_user.is_admin:
        flash('You do not have permission to delete questions.', 'error')
        return redirect(url_for('main.index'))
    
    question = Question.query.get_or_404(question_id)
    try:
        db.session.delete(question)
        db.session.commit()
        flash('Question deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting question: {str(e)}', 'error')
    
    return redirect(url_for('questions.admin_list_questions')) 