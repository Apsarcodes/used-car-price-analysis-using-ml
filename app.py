from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import sys
import traceback
from datetime import datetime

# Initialize Flask app with proper static file configuration
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Add the directory containing used_car_prediction.py to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize model metrics with default values
model_metrics = {
    'r2_score': 0.85,  # Default value, will be updated after model loads
    'model_name': 'Random Forest',  # Default model name
    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_samples': 0,
    'feature_count': 0
}

# Try to import and initialize the prediction model
try:
    import used_car_prediction as prediction_model
    print("Successfully imported prediction model")
    
    # Update model metrics from the imported module
    if hasattr(prediction_model, 'best_results'):
        model_metrics.update({
            'r2_score': round(prediction_model.best_results['r2'], 3),
            'model_name': prediction_model.best_model_name,
            'training_samples': len(prediction_model.X_train),
            'feature_count': len(prediction_model.feature_columns)
        })
except ImportError as e:
    print(f"Error importing prediction model: {str(e)}")
    traceback.print_exc()
    prediction_model = None

# Load dataset with error handling
try:
    df = pd.read_csv("used_cars_india.csv")
    print("Dataset loaded successfully!")
    print(f"Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    traceback.print_exc()
    df = pd.DataFrame()

@app.route('/')
def home():
    """Render the main page with model information"""
    return render_template('index.html', model=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from the frontend"""
    if prediction_model is None:
        return jsonify({
            'success': False,
            'message': 'Prediction model not available',
            'model_status': 'error'
        })
    
    try:
        data = request.form.to_dict()
        print(f"Received prediction request: {data}")
        
        # Validate required fields
        required_fields = ['brand', 'name', 'year', 'km_driven', 'fuel', 'transmission', 'owner']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            })
        
        # Convert numeric fields with validation
        try:
            year = int(data['year'])
            km_driven = int(data['km_driven'])
            
            # Validate reasonable ranges
            current_year = datetime.now().year
            if year < 1980 or year > current_year:
                return jsonify({
                    'success': False,
                    'message': f'Year must be between 1980 and {current_year}'
                })
                
            if km_driven < 0 or km_driven > 500000:
                return jsonify({
                    'success': False,
                    'message': 'KM Driven must be between 0 and 500,000'
                })
                
            data['year'] = year
            data['km_driven'] = km_driven
            
        except ValueError:
            return jsonify({
                'success': False,
                'message': 'Year and KM Driven must be valid numbers'
            })
        
        # Make prediction
        prediction = prediction_model.predict_car_price(data)
        
        if prediction is not None:
            return jsonify({
                'success': True,
                'prediction': round(prediction, 2),
                'prediction_lakhs': round(prediction / 100000, 2),
                'formatted_price': f'₹{round(prediction, 2):,}',
                'formatted_lakhs': f'₹{round(prediction / 100000, 2):,} lakhs'
            })
        return jsonify({
            'success': False,
            'message': 'Prediction returned None'
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Prediction failed: {str(e)}'
        })

@app.route('/get_options')
def get_options():
    """Provide dropdown options for the frontend"""
    if df.empty:
        return jsonify({
            'success': False,
            'message': 'Dataset not loaded'
        })
    
    try:
        # Get unique values for each category
        brands = sorted(df['Brand'].dropna().astype(str).unique().tolist())
        fuels = sorted(df['Fuel_Type'].dropna().astype(str).unique().tolist())
        transmissions = sorted(df['Transmission'].dropna().astype(str).unique().tolist())
        owners = sorted(df['Owner_Type'].dropna().astype(str).unique().tolist())
        
        # Get year range
        if 'Year' in df.columns:
            min_year = int(df['Year'].min())
            max_year = int(df['Year'].max())
        else:
            min_year = 2000
            max_year = datetime.now().year
        
        return jsonify({
            'success': True,
            'brands': brands,
            'fuels': fuels,
            'transmissions': transmissions,
            'owners': owners,
            'year_range': {
                'min': min_year,
                'max': max_year
            }
        })
    except Exception as e:
        print(f"Options error: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Failed to get options: {str(e)}'
        })

@app.route('/model_info')
def model_info():
    """Provide model information for the frontend"""
    return jsonify({
        'success': True,
        'model': model_metrics
    })

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({
        'status': 'active',
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_loaded': prediction_model is not None,
        'data_loaded': not df.empty
    })

if __name__ == '__main__':
    # Get port from environment variable or use 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the app on all network interfaces
    app.run(host='0.0.0.0', port=port, debug=True)