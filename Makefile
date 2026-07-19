#mcandrew

#--Set up environment variables
PYTHON ?= python3 -W ignore
R ?= Rscript

#--Set up virtual environment
VENV_DIR := .forecast
VENV_PYTHON := $(VENV_DIR)/bin/python -W ignore

full_forecast_pipeline: build_env data_pipeline forecast_component_models ensemble_time

#--Build environment----------------------------------------------------------------------------
build_env:
	@echo "build forecast environment"
	@$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install -r requirements.txt

#--Download data--------------------------------------------------------------------------------
data_pipeline: download_ili download_clinical_data download_hosp_pct_data download_target_data
download_clinical_data:
	@echo "Downloading Public lab data"
	@$(R) ./analysis_data/download_lab_percentage_data.R
	@$(VENV_PYTHON) ./analysis_data/format_lab_data.py

download_ili:
	@echo "Downloading recent ILINet data"
	@$(VENV_PYTHON) ./analysis_data/build_ili_data.py
	@$(VENV_PYTHON) ./analysis_data/format_ili_data.py

download_hosp_pct_data:
	@echo "Downloading NHSNpct hosp data"
	@$(VENV_PYTHON) ./analysis_data/download_percent_reported_hosps.py

download_target_data:
	@echo "Download target data"
	@$(R) ./data/target-data/get_target_data.R

#--REALTIME-------------------------------------------------------------------------------------
forecast_component_models:
	@echo "Running component model 1 in real-time"
	@$(VENV_PYTHON) ./run_real-time_shape_forecasts.py --real_time True
	@echo "Running component model 2 in real-time"
	@$(VENV_PYTHON) ./run_real-time_deriv_shape_forecasts.py --real_time True
	@echo "Running component model 3 in real-time"
	@$(VENV_PYTHON) ./run_real-time_KFmodel.py --real_time True

ensemble_time: score_models build_adaptive_ensemble

score_models:
	@echo "Scoring"
	@$(VENV_PYTHON) ./multi_model_ensemble_weights/combine_models.py
	@$(R) ./multi_model_ensemble_weights/evaluate_forecasts.R

build_adaptive_ensemble:
	@echo "Build Adaptive Ensemble"
	@$(VENV_PYTHON) ./multi_model_ensemble_weights/adaptive_wis_ensemble_optimal.py


#--These are only run once per season to collect seasonal shapes---------------------------------
generate_shapes:
	@echo "(This is run once) colect past shapes"
	@$(VENV_PYTHON) ./analysis/normalize_and_derive_B_weights.py

#--BUILD CONSTRAINTS
predict_total_hospitalizations:
	@echo "Predict total hospitalizations"
	@$(VENV_PYTHON) ./analysis/predict_total_US_hosps.py
	@$(VENV_PYTHON) ./analysis/from_total_hosps_to_state_hosps/regress.py


predict_peak_estimate:
	@echo "Predict A/B peaks"
	@$(VENV_PYTHON) ./analysis/







#--RETRO-------------------------------------------------------------------------------------
ensemble_time_retro: score_retro_models build_adaptive_ensemble_retro

score_retro_models:
	@echo "Scoring"
	@$(VENV_PYTHON) ./multi_model_ensemble_weights/retro/combine_shape_models.py
	@$(R) ./multi_model_ensemble_weights/retro/evaluate_forecats.R

#--Build Ensemble Weights
build_adaptive_ensemble_retro:
	@echo "Build Adaptive Ensemble"
	@$(VENV_PYTHON) ./multi_model_ensemble_weights/retro/adaptive_wis_ensemble_optiimal.py

