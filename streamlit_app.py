# streamlit_app_with_ui_cleaning_full.py
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime

# -----------------------------
# 1️⃣ LOAD MODELS & ARTIFACTS
# -----------------------------
MODEL_PATH = "models/"

models = {
    "Ensemble": joblib.load(MODEL_PATH + "ENSEMBLE_model.pkl"),
    "Random Forest (RF)": joblib.load(MODEL_PATH + "RF_model.pkl"),
    "XGBoost (XGB)": joblib.load(MODEL_PATH + "XGB_model.pkl"),
    "MLP Neural Network": joblib.load(MODEL_PATH + "MLP_model.pkl"),
    "Logistic Regression": joblib.load(MODEL_PATH + "LogReg_model.pkl"),
    "Support Vector Machine (SVM)": joblib.load(MODEL_PATH + "SVM_model.pkl"),
    "Multiple Linear Regression (MLR)": joblib.load(MODEL_PATH + "MMR_model.pkl")
}

feature_encoders = joblib.load(MODEL_PATH + "feature_encoders.pkl")
feature_scaler = joblib.load(MODEL_PATH + "feature_scaler.pkl")
label_encoder_target = joblib.load(MODEL_PATH + "label_encoder_target.pkl")
selected_features = joblib.load(MODEL_PATH + "selected_features.pkl")

df_cleaned = pd.read_csv(MODEL_PATH + "full_dataset_for_streamlit.csv", encoding="ISO-8859-1")
uni_df = pd.read_csv("Notebooks/Uni_Data.csv", encoding="ISO-8859-1")

df_cleaned.columns = df_cleaned.columns.str.strip()
uni_df.columns = uni_df.columns.str.strip()

career_map = {0: "Business", 1: "IT", 2: "Medicine", 3: "Engineering"}
career_fields = ["Business", "IT", "Medicine", "Engineering"]

if "Career_Interest" in df_cleaned.columns and df_cleaned["Career_Interest"].dtype != object:
    df_cleaned["Career_Interest"] = df_cleaned["Career_Interest"].map(career_map)

# -----------------------------
# 2️⃣ DETECT ONE-HOT COLUMNS
# -----------------------------
ui_base_features = [
    "Date_Of_Birth", "Gender", "Personality_Trait", "Aptitude", "Parents_Education",
    "Family_Lifestyle", "Location", "Math", "English", "Bio", "Chemistry", "Physics",
    "ICT", "Business", "if_HS_Student", "Study_Method", "Study_Habit",
    "English_Proficiency", "IELTS_Score", "Favorite_Subject", "Career_Interest",
    "Extracurricular", "Achievements", "Chosen_University", "Influence"
]

one_hot_cols = []
for c in df_cleaned.columns:
    if c in ui_base_features:
        continue
    col_vals = df_cleaned[c].dropna().unique()
    try:
        if set(np.unique(col_vals)).issubset({0, 1}):
            one_hot_cols.append(c)
    except Exception:
        continue
one_hot_cols = sorted(one_hot_cols)

extracurricular_options = [c for c in one_hot_cols if ("hack" in c.lower() or "program" in c.lower() or "coding" in c.lower() or "club" in c.lower() or "volunt" in c.lower() or "award" in c.lower() or "/" in c or " " in c)]
if not extracurricular_options:
    extracurricular_options = one_hot_cols.copy()
achievements_options = extracurricular_options.copy()

# -----------------------------
# 3️⃣ CLEANING FUNCTIONS
# -----------------------------
def clean_gender(g):
    if pd.isna(g):
        return np.nan
    g = str(g).strip().title()
    if g in ["Male", "Female", "Other"]:
        return g
    return np.nan

def parse_dob_to_age(dob_str):
    try:
        dob = pd.to_datetime(dob_str, dayfirst=True, errors='coerce')
        if pd.isna(dob):
            return np.nan
        today = pd.Timestamp.today()
        age = today.year - dob.year
        if (today.month, today.day) < (dob.month, dob.day):
            age -= 1
        if age < 10 or age > 30:
            return np.nan
        return int(age)
    except Exception:
        return np.nan

def age_to_age_group(age):
    try:
        if pd.isna(age):
            return "15-to-20"
        age = int(age)
        if 10 <= age <= 15:
            return "10-to-15"
        if 15 < age <= 20:
            return "15-to-20"
        return "20-to-30"
    except Exception:
        return "15-to-20"

def encode_parents_education(v):
    if isinstance(v, str) and v.strip().lower().startswith("g"):
        return 1
    try:
        return int(v)
    except Exception:
        return 0

def map_family_lifestyle(v):
    if pd.isna(v):
        return np.nan
    v = str(v).strip().title()
    mapping = {'Limited / Basic':0,'Moderate / Average':1,'Comfortable / Above Average':2}
    for k in mapping:
        if v.lower() == k.lower():
            return mapping[k]
    if "limited" in v.lower():
        return 0
    if "moderate" in v.lower() or "average" in v.lower():
        return 1
    if "comfortable" in v.lower() or "above" in v.lower():
        return 2
    return np.nan

def clean_score(val):
    try:
        if pd.isna(val) or str(val).strip() == "":
            return 0
        v = float(val)
        if np.isnan(v):
            return 0
        return v
    except Exception:
        return 0

def map_influence(v):
    internal = {"personal interest / passion", "previous academic performance / aptitude"}
    if pd.isna(v):
        return "External"
    s = str(v).strip().lower()
    if s in internal:
        return "Internal"
    return "External"

# Precompute fallback defaults
gender_mode_by_career = {}
family_mode_by_career = {}
if 'Career_Interest' in df_cleaned.columns:
    for c in career_fields:
        sub = df_cleaned[df_cleaned['Career_Interest'] == c]
        if 'Gender' in sub.columns:
            try: gender_mode_by_career[c] = sub['Gender'].mode().iloc[0]
            except: gender_mode_by_career[c] = np.nan
        if 'Family_Lifestyle_Encoded' in sub.columns:
            try: family_mode_by_career[c] = sub['Family_Lifestyle_Encoded'].mode().iloc[0]
            except: family_mode_by_career[c] = np.nan
global_gender_mode = df_cleaned['Gender'].mode().iloc[0] if 'Gender' in df_cleaned.columns else "Male"
global_family_mode = df_cleaned['Family_Lifestyle_Encoded'].mode().iloc[0] if 'Family_Lifestyle_Encoded' in df_cleaned.columns else 1

subject_cols = ['Math','English','Bio','Chemistry','Physics','ICT','Business','IELTS_Score']
subject_medians = {}
for col in subject_cols:
    if col in df_cleaned.columns:
        non_zero = df_cleaned[df_cleaned[col] != 0][col]
        subject_medians[col] = non_zero.median() if not non_zero.empty else 0
    else:
        subject_medians[col] = 0

# -----------------------------
# 4️⃣ MAIN CLEANING FUNCTION
# -----------------------------
def clean_ui_input(raw: dict):
    cleaned = {}
    dob = raw.get("Date_Of_Birth","")
    age = parse_dob_to_age(dob)
    if pd.isna(age):
        age = raw.get("Age", np.nan)
    if pd.isna(age):
        try: age = int(df_cleaned['Age'].median())
        except: age = 16
    cleaned['Age'] = int(age)
    cleaned['Age_Group'] = age_to_age_group(age)

    g = clean_gender(raw.get("Gender", np.nan))
    desired_career = raw.get("Career_Interest", np.nan)
    if pd.isna(g):
        if desired_career in career_fields and gender_mode_by_career.get(desired_career):
            g = gender_mode_by_career[desired_career]
        else:
            g = global_gender_mode
    cleaned['Gender'] = g

    cleaned['Personality_Trait'] = raw.get('Personality_Trait','')
    try: cleaned['Aptitude'] = np.clip(int(raw.get('Aptitude',0)),0,10)
    except: cleaned['Aptitude'] = 0
    cleaned['Parents_Education_Encoded'] = encode_parents_education(raw.get('Parents_Education',0))

    fam = map_family_lifestyle(raw.get("Family_Lifestyle", np.nan))
    if pd.isna(fam):
        if desired_career in career_fields and family_mode_by_career.get(desired_career):
            fam = family_mode_by_career[desired_career]
        else:
            fam = global_family_mode
    cleaned['Family_Lifestyle_Encoded'] = int(fam)

    cleaned['Address'] = raw.get('Location', raw.get('Address',''))

    for s in subject_cols:
        val = clean_score(raw.get(s,0))
        if val < 0: val = 0
        if val > 1000: val = subject_medians.get(s,0)
        cleaned[s] = val

    scores_for_avg = [cleaned[c] for c in ['Math','English','Bio','Chemistry','Physics','ICT','Business'] if cleaned.get(c,0)!=0]
    cleaned['Average_Score'] = round(float(np.mean(scores_for_avg)) if scores_for_avg else float(df_cleaned['Average_Score'].median() if 'Average_Score' in df_cleaned.columns else 50),2)

    val_hs = raw.get('if_HS_Student', False)
    cleaned['if_HS_Student'] = 1 if str(val_hs).strip().lower() in ['1','yes','true','y','t'] else 0
    cleaned['Study_Method'] = raw.get('Study_Method','')
    cleaned['Study_Habit'] = raw.get('Study_Habit','')
    cleaned['English_Proficiency'] = raw.get('English_Proficiency','')
    cleaned['IELTS_Score'] = cleaned.get('IELTS_Score', cleaned.get('IELTS_Score',0))
    cleaned['Favorite_Subject'] = raw.get('Favorite_Subject','')
    cleaned['Career_Interest'] = raw.get('Career_Interest','')
    cleaned['Chosen_University'] = raw.get('Chosen_University','')
    cleaned['Influence'] = map_influence(raw.get('Influence',''))

    # One-hot extracurriculars
    extracurricular_selected = [str(x) for x in raw.get('Extracurricular',[]) or []]
    achievements_selected = [str(x) for x in raw.get('Achievements',[]) or []]
    for col in one_hot_cols:
        cleaned[col] = 1 if col in extracurricular_selected or col in achievements_selected else 0

    return cleaned

# -----------------------------
# 5️⃣ PREPARE INPUT FOR MODEL
# -----------------------------
def prepare_input_for_model(cleaned_dict):
    all_features = list(feature_scaler.feature_names_in_)
    df_input = pd.DataFrame(columns=all_features)
    for col in df_input.columns:
        df_input.at[0,col] = np.nan
    for k,v in cleaned_dict.items():
        if k in df_input.columns:
            df_input.at[0,k] = v
    for col in df_input.columns:
        if pd.isna(df_input.at[0,col]):
            df_input.at[0,col] = 0 if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]) else ""
    for col, enc in feature_encoders.items():
        if col in df_input.columns:
            try:
                raw_val = df_input.at[0,col]
                df_input.at[0,col] = enc.transform([raw_val])[0] if raw_val in enc.classes_ else -1
            except: df_input.at[0,col] = -1
    for c in df_input.columns:
        try: df_input[c] = pd.to_numeric(df_input[c])
        except: pass
    X_scaled = feature_scaler.transform(df_input)
    X_df = pd.DataFrame(X_scaled, columns=df_input.columns)
    X_final = X_df.loc[:, selected_features]

# 🚨 Absolute safety guard: remove any NaN still left
    X_final = X_final.fillna(0)

    return X_final


# -----------------------------
# 6️⃣ PREDICTION
# -----------------------------
def predict_with_model(model,X):
    pred = model.predict(X)
    if model.__class__.__name__=="LinearRegression":
        pred = np.rint(pred).astype(int)
        pred = np.clip(pred,0,len(career_map)-1)
    return pred

# -----------------------------
# -----------------------------
# 7️⃣ GAP SUGGESTIONS (Updated)
# -----------------------------
career_features_map = {
    "Business": ['Business', 'Average_Score', 'Math', 'Study_Method', 'English_Proficiency', 'Favorite_Subject'],
    "IT": ['Math', 'Average_Score', 'Programming / coding clubs', 'Hackathons / App development projects',
           'Building miniatures / models', 'Programming / Coding award'],
    "Medicine": ['Bio', 'Chemistry', 'Physics', 'Average_Score',
                 'Volunteering at hospitals, clinics, or NGOs'],
    "Engineering": ['Math', 'Physics', 'Chemistry', 'Average_Score',
                    'Model building recognition', 'Building miniatures / models']
}

career_extracurricular_map = {
    "Business": [c for c in extracurricular_options if c in career_features_map["Business"]],
    "IT": [c for c in extracurricular_options if c in career_features_map["IT"]],
    "Medicine": [c for c in extracurricular_options if c in career_features_map["Medicine"]],
    "Engineering": [c for c in extracurricular_options if c in career_features_map["Engineering"]],
}

def compute_grouped_gap_suggestions_local(df, student_data, career_label):
    if 'Career_Interest' not in df.columns:
        return ["No career grouping available in training data."]
    
    group = df[df['Career_Interest'] == career_label]
    if group.empty:
        return ["No training data for this career to compare."]
    
    # Start with features mapped for this career
    features = career_features_map.get(
        career_label,
        ['Math','English','Bio','Chemistry','Physics','ICT','Business','Average_Score']
    )
    
    # Include all one-hot columns present in student_data
    features = [f for f in features if f in student_data] + \
               [f for f in one_hot_cols if f in student_data and f not in features]
    
    if not features:
        return ["No matching features to compare."]
    
    suggestions = []

    # ----- NUMERIC & CATEGORICAL FEATURES -----
    for f in features:
        if f not in student_data:
            continue

        student_val = student_data.get(f, None)
        group_vals = group[f].dropna() if f in group.columns else pd.Series(dtype=float)
        if group_vals.empty:
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(group_vals):
            try:
                gap = group_vals.mean() - float(student_val)
                if gap > 5:
                    suggestions.append(f"Your {f} score is below the average of peers. Consider improving in this area.")
                elif gap < -5:
                    suggestions.append(f"Your {f} score is above the average of peers. Keep it up!")
            except:
                continue

        # Categorical / string
        elif not set(group_vals.unique()).issubset({0,1}):
            modes = [str(x).lower() for x in group_vals.mode().values]
            if str(student_val).lower() not in modes:
                common_mode = group_vals.mode().values[0] if not group_vals.mode().empty else "N/A"
                suggestions.append(f"Consider exploring **{f}** options aligned with peers (common: {common_mode}).")

    # ----- ONE-HOT / EXTRACURRICULARS -----
    relevant_activities = career_extracurricular_map.get(career_label, [])
    if relevant_activities:
        missing_activities = [act for act in relevant_activities if student_data.get(act, 0) == 0]
        if missing_activities:
            suggestions.append(f"Consider participating in: {', '.join(missing_activities)} (peers often do).")
        else:
            suggestions.append("Great! You have participated in all relevant activities for this career.")

    if not suggestions:
        suggestions = ["Your profile matches well with this career."]

    return suggestions






# -----------------------------
# 8️⃣ UNIVERSITY RECOMMENDATIONS
# -----------------------------
def recommend_universities(student, country, career):
    df = uni_df[(uni_df["Country"] == country) & (uni_df["Field of Study"] == career)]
    if df.empty:
        return ["No universities found."]

    # Compute GPA fit
    def score_uni(row):
        score = 0
        try:
            if pd.notna(row.get("GPA")):
                uni_gpa_percent = row["GPA"] / 4 * 100
                if student["Average_Score"] >= uni_gpa_percent:
                    score += 1
        except:
            pass
        return score

    df["Fit"] = df.apply(score_uni, axis=1)

    # Convert QS Ranking to numeric for sorting
    def parse_qs_rank(rank_str):
        try:
            if pd.isna(rank_str):
                return 9999
            rank_str = str(rank_str)
            # Extract first number if range like "1001-1200" or "~751-800"
            rank = re.findall(r'\d+', rank_str)
            if rank:
                return int(rank[0])
            else:
                return 9999
        except:
            return 9999

    df["QS_Rank_Num"] = df["World Ranking"].apply(parse_qs_rank)

    # Sort by QS rank (ascending), then Fit score (descending)
    df_sorted = df.sort_values(by=["QS_Rank_Num", "Fit"], ascending=[True, False]).head(5)

    # Add warning if GPA not met
    df_sorted['Warning'] = ""
    for i, row in df_sorted.iterrows():
        if pd.notna(row.get("GPA")):
            uni_gpa_percent = row["GPA"] / 4 * 100
            if student["Average_Score"] < uni_gpa_percent:
                df_sorted.at[i, 'Warning'] = "⚠️ You may not meet the GPA requirement. Try to improve your scores."

    return df_sorted


def escape_md(text):
    if text is None: return ""
    return re.sub(r'([\\`*_{}[\]()#+\-.!])', r'\\\1', str(text))

# -----------------------------
# 9️⃣ STREAMLIT UI
# -----------------------------
st.title("🎓 Novis AI - Career & University Guidance System")

# Subject numeric inputs
st.header("Academic / Scores")
cols = st.columns(4)
subj_defaults = {'Math':50,'English':50,'Bio':0,'Chemistry':0,'Physics':0,'ICT':0,'Business':0}
student_vals = {}
for i,s in enumerate(['Math','English','Bio','Chemistry','Physics','ICT','Business']):
    student_vals[s] = cols[i%4].number_input(f"{s}",0.0,1000.0,float(subj_defaults[s]),step=1.0,key=f"num_{s}")

# Personal & Demographics
st.header("Personal / Demographics")
dob_input = st.text_input("Date_Of_Birth (dd/mm/YYYY)", value="19/01/1997")
gender_input = st.selectbox("Gender", ["Male","Female","Other"])
mbti_choices = ["INTJ","INTP","ENTJ","ENTP","INFJ","INFP","ENFJ","ENFP","ISTJ","ISFJ","ESTJ","ESFJ","ISTP","ISFP","ESTP","ESFP"]
personality_input = st.selectbox("Personality_Trait (MBTI)", mbti_choices)
aptitude_input = st.slider("Aptitude (0–10)",0,10,5)
parents_edu_input = st.selectbox("Parents_Education", ["Graduate","Non-Graduate"])
family_input = st.selectbox("Family_Lifestyle", ["Comfortable / Above Average","Moderate / Average","Limited / Basic"])
state_map = {
    1: "Yangon",
    2: "Mandalay",
    3: "Naypyitaw",
    4: "Bago"
    # add all your states/regions here
}


# Convert numeric codes to names
location_options = sorted(df_cleaned['Address'].dropna().astype(int).map(state_map).dropna().unique().tolist())

location_input = st.selectbox("Location (State/Region)", location_options)

hs_input = st.radio("Are you a high-school student?", ["Yes","No"])
study_method_input = st.selectbox("Study_Method", ["Self-study","Group study","Tuition / Extra Classes"])
study_habit_input = st.selectbox("Study_Habit", ["<1h","1–2h","2–3h","3–4h",">4h"])
english_prof_input = st.selectbox("English_Proficiency", ["Beginner","Intermediate","Advanced"])
ielts_input = st.number_input("IELTS_Score (if any)", min_value=0.0,max_value=9.0,step=0.5,value=0.0)
favorite_subject_input = st.selectbox("Favorite_Subject", ["Math","English","Bio","Chemistry","Physics","ICT","Business"])

extracurricular_selected = st.multiselect("Extracurricular", options=extracurricular_options)
achievements_selected = st.multiselect("Achievements", options=achievements_options)
chosen_uni_input = st.text_input("Chosen_University (if any)")
influence_input = st.selectbox("Influence", [
    "Personal interest / passion","Family influence / parental guidance","Teachers / mentors",
    "Previous academic performance / aptitude","Job availability / employment prospects",
    "Peer influence / friends","Other"
])

student_raw = {
    "Date_Of_Birth": dob_input,
    "Gender": gender_input,
    "Personality_Trait": personality_input,
    "Aptitude": aptitude_input,
    "Parents_Education": parents_edu_input,
    "Family_Lifestyle": family_input,
    "Location": location_input,
    **student_vals,
    "if_HS_Student": 1 if hs_input=="Yes" else 0,
    "Study_Method": study_method_input,
    "Study_Habit": study_habit_input,
    "English_Proficiency": english_prof_input,
    "IELTS_Score": ielts_input,
    "Favorite_Subject": favorite_subject_input,

    "Extracurricular": extracurricular_selected,
    "Achievements": achievements_selected,
    "Chosen_University": chosen_uni_input,
    "Influence": influence_input
}

mode = st.radio("Choose Mode", ["Model Predicts Career", "User Chooses Career"])
desired_career = None
if mode == "User Chooses Career":
    desired_career = st.selectbox("Select Career Goal", career_fields)
else:
    model_choice = st.selectbox("Select ML Model", list(models.keys()), index=0)


country = st.selectbox("Select Country for University Guidance", sorted(uni_df["Country"].dropna().unique()))

if st.button("Get Guidance"):
    cleaned = clean_ui_input(student_raw)
    if mode == "User Chooses Career" and desired_career:
        cleaned['Career_Interest'] = desired_career

    X_ready = prepare_input_for_model(cleaned)

    if mode == "Model Predicts Career":
        pred = predict_with_model(models[model_choice], X_ready)
        pred_numeric = int(pred[0])
        predicted_career = career_map.get(pred_numeric, "Unknown")
        st.subheader(f"Predicted Career ({model_choice})")
    else:
        predicted_career = desired_career
        st.subheader(f"User Career Goal: {predicted_career}")

    st.write("**Career / Field:**", predicted_career)

    suggestions = compute_grouped_gap_suggestions_local(df_cleaned, cleaned, predicted_career)
    st.subheader("📊 Gap Suggestions / Recommendations")
    for s in suggestions:
        st.write("-", escape_md(s))

    st.subheader(f"🏛 Recommended Universities in {country} for {predicted_career}")
    unis = recommend_universities(cleaned, country, predicted_career)
    if isinstance(unis, list) and len(unis) == 1:
        st.warning(unis[0])
    else:
        for _, r in unis.iterrows():
            st.write("University Name:", escape_md(r.get("University Name", "")))
            st.write("Program:", escape_md(r.get("Program Name", "")))
            st.write("Field:", escape_md(r.get("Field of Study", "")))
            st.write("GPA Required:", r.get("GPA", "N/A"))
            st.write("URL:", escape_md(r.get("Source URL", "")))
            if r.get("Warning"):
                st.warning(r.get("Warning"))
            st.write("---")
