# %% Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import dice_ml
from dice_ml import Data, Model

# Custom DataLoader (Replace with your actual data handling logic)
class DataLoader:
    def __init__(self):
        self.data = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
    
    def load_dataset(self):
        from sklearn.datasets import fetch_openml
        # Example dataset: Fetch Stroke Dataset
        self.data = fetch_openml(data_id=534, as_frame=True)  # Replace with your dataset
    
    def preprocess_data(self):
        # Preprocessing: Dummy encoding categorical variables and splitting data
        target = 'stroke'
        self.data[target] = self.data[target].astype(int)
        self.data = self.data.dropna()  # Drop missing values for simplicity
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)
    
    def get_data_split(self):
        from sklearn.model_selection import train_test_split
        target = 'stroke'
        X = self.data.drop(columns=[target])
        y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def oversample(self, X_train, y_train):
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        return sm.fit_resample(X_train, y_train)

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
X_train, X_test, y_train, y_test = data_loader.get_data_split()
X_train, y_train = data_loader.oversample(X_train, y_train)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# %% Fit blackbox model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# %% Create diverse counterfactual explanations
# Initialize DiCE data and model
data_dice = Data(
    dataframe=data_loader.data,
    continuous_features=['age', 'avg_glucose_level', 'bmi'],  # Specify continuous features
    outcome_name='stroke'  # Target column
)
rf_dice = Model(model=rf, backend="sklearn")

# Initialize DiCE explainer
explainer = dice_ml.Dice(
    data=data_dice,
    model=rf_dice,
    method="random"  # Counterfactual generation method
)

# %% Generate Counterfactual Explanations
input_datapoint = X_test.iloc[0:1]  # Single input point
cf = explainer.generate_counterfactuals(
    input_datapoint,
    total_CFs=3,
    desired_class="opposite"
)
cf.visualize_as_dataframe(show_only_changes=True)

# %% Create feasible (conditional) Counterfactuals
features_to_vary = ['avg_glucose_level', 'bmi', 'smoking_status_smokes']
permitted_range = {
    'avg_glucose_level': [50, 250],
    'bmi': [18, 35]
}
cf = explainer.generate_counterfactuals(
    input_datapoint,
    total_CFs=3,
    desired_class="opposite",
    permitted_range=permitted_range,
    features_to_vary=features_to_vary
)
cf.visualize_as_dataframe(show_only_changes=True)
