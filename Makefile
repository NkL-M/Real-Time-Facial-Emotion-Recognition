# --------------------- #
#      MAIN ACTIONS     #
# --------------------- #
installation:
	@pip install -e .
	@pip install -r requirements.txt
	@echo "Intallation done."

app_demo:
	@echo "Running facial emotion recognition demo..."
	-@streamlit run emotion_recognition/app/streamlit_app.py

run_local_fer:
	@echo "Running local facial emotion recognition on webcam..."
	python -c 'from emotion_recognition.interface.main import main; main()'

# ------------- #
#     TESTS     #
# ------------- #

# default: pytest

# default: pylint pytest

# pylint:
# 	find . -iname "*.py" -not -path "./tests/test_*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

# pytest:
# 	echo "no tests"

# tests:
# 	@echo "🧪 Running tests..."
# 	@pytest -q --disable-warnings --maxfail=1
# 	@echo "Tests completed."

# tests:
# 	@echo "🧪 Running tests with coverage..."
# 	@pytest --cov=emotion_recognition --cov-report=term-missing --disable-warnings --maxfail=1
# 	@echo "✨ Coverage report generated."

# --------------- #
#     CLEANING    #
# --------------- #
cleaning:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@rm -f **/.ipynb_checkpoints
	@rm -fr **/__pycache__ **/*.pyc
	@rm -rf build dist
	@rm -rf *.egg-info
	@rm -rf *.dist-info
	@rm -rf proj.egg-info
	@rm -rf proj-*.dist-info
	@echo "Cleaned."
