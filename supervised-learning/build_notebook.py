from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
import os

OUT = "./ml_practice_notebook.pdf"

# ── Colour palette ────────────────────────────────────────────────────────────
COVERS = [
    colors.HexColor("#4C78A8"),  # Section 1 – Visualizations  (blue)
    colors.HexColor("#1D9E75"),  # Section 2 – Regression plan (teal)
    colors.HexColor("#E45756"),  # Section 3 – Datasets map    (coral)
    colors.HexColor("#BA7517"),  # Section 4 – Industry std    (amber)
    colors.HexColor("#7F77DD"),  # Section 5 – Classification  (purple)
]
LIGHT = [
    colors.HexColor("#E6F1FB"),
    colors.HexColor("#E1F5EE"),
    colors.HexColor("#FAECE7"),
    colors.HexColor("#FAEEDA"),
    colors.HexColor("#EEEDFE"),
]
RULE_CLR = [
    colors.HexColor("#B5D4F4"),
    colors.HexColor("#9FE1CB"),
    colors.HexColor("#F5C4B3"),
    colors.HexColor("#FAC775"),
    colors.HexColor("#CECBF6"),
]
TAG_BG = [
    colors.HexColor("#378ADD"),
    colors.HexColor("#0F6E56"),
    colors.HexColor("#993C1D"),
    colors.HexColor("#854F0B"),
    colors.HexColor("#534AB7"),
]

W, H = A4
MARGIN = 18*mm

# ── Style factory ─────────────────────────────────────────────────────────────
def make_styles(sec):
    accent = COVERS[sec]
    light  = LIGHT[sec]
    return {
        "sec_title": ParagraphStyle("sec_title",
            fontSize=26, leading=32, textColor=colors.white,
            fontName="Helvetica-Bold", alignment=TA_CENTER),
        "sec_sub": ParagraphStyle("sec_sub",
            fontSize=13, leading=18, textColor=colors.HexColor("#EAEAEA"),
            fontName="Helvetica", alignment=TA_CENTER),
        "h1": ParagraphStyle("h1",
            fontSize=16, leading=22, textColor=accent,
            fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=14),
        "h2": ParagraphStyle("h2",
            fontSize=13, leading=18, textColor=accent,
            fontName="Helvetica-Bold", spaceAfter=3, spaceBefore=10),
        "h3": ParagraphStyle("h3",
            fontSize=11, leading=16, textColor=colors.HexColor("#333333"),
            fontName="Helvetica-Bold", spaceAfter=2, spaceBefore=6),
        "body": ParagraphStyle("body",
            fontSize=10, leading=15, textColor=colors.HexColor("#222222"),
            fontName="Helvetica"),
        "bullet": ParagraphStyle("bullet",
            fontSize=10, leading=15, textColor=colors.HexColor("#222222"),
            fontName="Helvetica", leftIndent=14, bulletIndent=4),
        "code": ParagraphStyle("code",
            fontSize=9, leading=14, textColor=colors.HexColor("#1A1A2E"),
            fontName="Courier", backColor=colors.HexColor("#F4F4F8"),
            leftIndent=10, rightIndent=10, borderPad=4,
            borderColor=colors.HexColor("#CCCCDD"), borderWidth=0.5),
        "q": ParagraphStyle("q",
            fontSize=10, leading=15, textColor=colors.HexColor("#444444"),
            fontName="Helvetica-Oblique", leftIndent=12,
            borderLeftColor=accent, borderLeftWidth=3, borderLeftPadding=8),
        "tag": ParagraphStyle("tag",
            fontSize=9, leading=13, textColor=colors.white,
            fontName="Helvetica-Bold", alignment=TA_CENTER),
        "rule_label": ParagraphStyle("rule_label",
            fontSize=9, leading=13, textColor=colors.HexColor("#555555"),
            fontName="Helvetica-Oblique"),
    }

# ── Cover page for each section ───────────────────────────────────────────────
def section_cover(story, sec_idx, emoji_char, title, subtitle):
    accent = COVERS[sec_idx]
    light  = LIGHT[sec_idx]
    st = make_styles(sec_idx)

    # coloured banner table
    banner = Table([[Paragraph(f"<b>{title}</b>", st["sec_title"])],
                    [Paragraph(subtitle, st["sec_sub"])]],
                   colWidths=[W - 2*MARGIN])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), accent),
        ("ROUNDEDCORNERS", [10]),
        ("TOPPADDING",    (0,0), (-1,-1), 22),
        ("BOTTOMPADDING", (0,0), (-1,-1), 22),
        ("LEFTPADDING",   (0,0), (-1,-1), 16),
        ("RIGHTPADDING",  (0,0), (-1,-1), 16),
    ]))
    story.append(banner)
    story.append(Spacer(1, 8))

def hr(story, sec_idx):
    story.append(HRFlowable(width="100%", thickness=1,
                             color=RULE_CLR[sec_idx], spaceAfter=4, spaceBefore=4))

def tag_pill(text, sec_idx):
    bg = TAG_BG[sec_idx]
    return Table([[Paragraph(f"<b>{text}</b>",
                             ParagraphStyle("tp", fontSize=8, leading=12,
                                            textColor=colors.white,
                                            fontName="Helvetica-Bold"))]],
                 colWidths=[len(text)*6 + 16])

def info_box(story, items, sec_idx, header=None):
    """Coloured info box with bullet list."""
    light = LIGHT[sec_idx]
    accent = COVERS[sec_idx]
    st = make_styles(sec_idx)
    rows = []
    if header:
        rows.append([Paragraph(f"<b>{header}</b>",
                               ParagraphStyle("bh", fontSize=10, leading=14,
                                              textColor=accent,
                                              fontName="Helvetica-Bold"))])
    for item in items:
        rows.append([Paragraph(f"&#x2022;&#160;{item}", st["bullet"])])
    t = Table(rows, colWidths=[W - 2*MARGIN - 8])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), light),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("LINEAFTER",     (0,0), (0,-1), 3, accent),
        ("ROUNDEDCORNERS",[6]),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

def two_col_table(story, rows_data, headers, sec_idx):
    """Two-column situation/plot table."""
    accent = COVERS[sec_idx]
    light  = LIGHT[sec_idx]
    cw = [(W - 2*MARGIN)*0.52, (W - 2*MARGIN)*0.48]
    tdata = [[Paragraph(f"<b>{h}</b>",
                        ParagraphStyle("th", fontSize=9, leading=13,
                                       textColor=colors.white,
                                       fontName="Helvetica-Bold"))
              for h in headers]]
    for r in rows_data:
        tdata.append([Paragraph(str(c),
                                ParagraphStyle("td", fontSize=9, leading=13,
                                               textColor=colors.HexColor("#222222"),
                                               fontName="Helvetica"))
                      for c in r])
    t = Table(tdata, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  accent),
        ("BACKGROUND",    (0,1), (-1,-1), light),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [light, colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.4, RULE_CLR[sec_idx]),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1  ─  Visualization Plots Guide
# ═══════════════════════════════════════════════════════════════════════════════
def build_section1(story):
    S = 0
    st = make_styles(S)
    section_cover(story, S, "", "Visualization Plots Guide",
                  "Which plot to use at every ML phase — EDA through production monitoring")
    story.append(Spacer(1, 10))

    phases = [
        ("1  Problem Understanding", [
            ("Regression target (SalePrice)", "Histogram / KDE", "See skewness, long tails, multimodality"),
            ("Classification target (Titanic)", "Count plot / bar chart", "Check class imbalance"),
            ("Target has extreme values", "Boxplot / violin plot", "Detect outliers"),
            ("Target changes over time", "Line plot", "See trend, seasonality, drift"),
        ]),
        ("2  Missing Data Analysis", [
            ("Many columns missing", "Missingness bar plot", "Shows which columns need attention"),
            ("Missingness related to target", "Boxplot target vs indicator", "Tests if missingness carries signal"),
            ("Missing values co-occur", "Missingness heatmap", "Patterns across rows/columns"),
            ("Missingness by group", "Grouped bar plot", "Example: missing basement by house type"),
        ]),
        ("3  Univariate Feature Analysis", [
            ("Numeric feature", "Histogram / KDE", "Distribution shape"),
            ("Numeric with outliers", "Boxplot", "Detect extreme values"),
            ("Categorical feature", "Count plot", "Cardinality and rare categories"),
            ("Ordinal feature", "Bar plot", "Check ordered relationship"),
        ]),
        ("4  Feature vs Target Analysis", [
            ("Numeric vs regression target", "Scatter plot", "Shows linear/nonlinear relationship"),
            ("Numeric vs target with noise", "Scatter + LOWESS line", "Reveals curve/trend"),
            ("Categorical vs regression target", "Boxplot / violin plot", "Compares target by category"),
            ("Categorical vs classification target", "Bar plot of target rate", "Shows risk/default rate"),
        ]),
        ("5  Correlation & Multicollinearity", [
            ("Many numeric features", "Correlation heatmap", "Finds related variables"),
            ("Strongest target relationships", "Sorted correlation bar plot", "Ranks useful features"),
            ("Linear model unstable", "VIF table / heatmap", "Detects multicollinearity"),
            ("Highly related predictors", "Pairplot on top features", "Understand feature clusters"),
        ]),
        ("6  Outlier Analysis", [
            ("Numeric feature extreme values", "Boxplot", "Detect feature outliers"),
            ("Outlier affects target", "Scatter plot", "See influential observations"),
            ("Model residuals extreme", "Residual plot", "Detect bad predictions"),
            ("Need robust model", "Actual vs predicted + labels", "See if Huber/RANSAC helps"),
        ]),
        ("7  Feature Engineering / Nonlinearity", [
            ("Curved relationship", "Scatter + smooth curve", "Suggests polynomial/spline"),
            ("Long right tail feature", "Histogram before/after log", "Check if log transform helps"),
            ("Two features affect target jointly", "2D scatter colored by target", "Shows interaction"),
            ("Nonlinear model performs better", "Partial dependence plot", "Shows learned nonlinear effect"),
        ]),
        ("8  Preprocessing Comparison", [
            ("Compare imputation strategies", "Boxplot of CV scores", "Shows stability across folds"),
            ("Check imputed values", "Histogram original vs imputed", "Ensures realistic values"),
            ("Compare scaling effect", "Boxplot before/after scaling", "Confirms robust scaling"),
            ("Encoding creates many columns", "Feature count bar chart", "Tracks dimensionality"),
        ]),
        ("9  Algorithm Comparison", [
            ("Regression models", "Bar plot of RMSE/MAE/R2", "RMSE, MAE, R2"),
            ("Classification models", "Bar plot of AUC/F1/Recall", "ROC AUC, F1, Recall"),
            ("Need stable model", "Error bars on CV scores", "Mean + std deviation"),
            ("Training vs validation gap", "Line/bar train vs CV score", "Detect overfitting"),
        ]),
        ("10  Hyperparameter Tuning", [
            ("One hyperparameter", "Line plot", "Shows best range"),
            ("Two hyperparameters", "Heatmap", "Shows interaction"),
            ("Random/Grid search", "Parallel coordinates plot", "See good parameter regions"),
            ("Overfitting check", "Train vs validation curve", "Detect too much complexity"),
        ]),
        ("11  Model Evaluation: Regression", [
            ("Overall prediction quality", "Actual vs predicted plot", "Perfect model lies on diagonal"),
            ("Error pattern", "Residuals vs predicted", "Detect bias/nonlinearity"),
            ("Residual distribution", "Histogram / KDE of residuals", "Check normality/skew"),
            ("Unequal error by price range", "Residuals vs actual", "Detect heteroscedasticity"),
        ]),
        ("12  Model Evaluation: Classification", [
            ("Basic errors", "Confusion matrix", "TP, FP, TN, FN"),
            ("Ranking quality", "ROC curve", "General discrimination"),
            ("Imbalanced data", "Precision-recall curve", "Better than ROC for rare positives"),
            ("Probability quality", "Calibration curve", "Are probabilities trustworthy?"),
        ]),
        ("13  Model Interpretation", [
            ("Linear model", "Coefficient plot", "Direction and strength"),
            ("Tree model", "Feature importance bar plot", "Important predictors"),
            ("Need reliable importance", "Permutation importance", "Model-agnostic"),
            ("Explain individual prediction", "SHAP waterfall/force plot", "Local explanation"),
        ]),
        ("14  Error Analysis", [
            ("Find bad predictions", "Top residual table", "Inspect failure cases"),
            ("Errors by category", "Boxplot residual by category", "Detect group bias"),
            ("Errors by target range", "Residuals vs actual", "See if expensive homes are underpredicted"),
            ("Spatial/location features", "Map or grouped residual plot", "Detect location bias"),
        ]),
        ("15  Production Monitoring", [
            ("Incoming missing values changed", "Missing-rate drift bar plot", "Data quality monitoring"),
            ("Feature distributions changed", "Train vs current histogram", "Detect feature drift"),
            ("Categorical levels changed", "New category count plot", "Detect schema drift"),
            ("Prediction distribution changed", "Histogram over time", "Detect model behavior shift"),
        ]),
    ]

    for phase_title, rows in phases:
        story.append(Paragraph(phase_title, st["h2"]))
        two_col_table(story, rows,
                      ["Situation", "Useful Plot / Why"], S)

    # Quick decision guide
    story.append(Spacer(1, 4))
    hr(story, S)
    story.append(Paragraph("Quick Decision Guide", st["h1"]))
    guide = [
        ("What does my target look like?", "Histogram / count plot"),
        ("Are there missing values?", "Missingness bar / heatmap"),
        ("Are there outliers?", "Boxplot / scatter plot"),
        ("Is relationship linear?", "Scatter + smooth line"),
        ("Are features correlated?", "Correlation heatmap"),
        ("Which model is best?", "CV metric bar plot with error bars"),
        ("Is model overfitting?", "Train vs validation curve"),
        ("Where does model fail?", "Residual plot / confusion matrix"),
        ("Why did model predict this?", "Feature importance / SHAP / PDP"),
        ("Is production data changing?", "Drift plots"),
    ]
    two_col_table(story, guide, ["Question", "Plot to Use"], S)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2  ─  Regression Learning Plan
# ═══════════════════════════════════════════════════════════════════════════════
def build_section2(story):
    S = 1
    st = make_styles(S)
    story.append(PageBreak())
    section_cover(story, S, "", "Regression Learning Plan",
                  "10-phase constructive path from OLS basics to production uncertainty")
    story.append(Spacer(1, 10))

    story.append(Paragraph("Core Learning Loop", st["h1"]))
    story.append(Paragraph(
        "For every regression topic: <b>Concept &#8594; small experiment &#8594; "
        "visual diagnosis &#8594; model comparison &#8594; error analysis &#8594; production concern</b>",
        st["body"]))
    story.append(Spacer(1, 6))

    phases = [
        ("Phase 1", "Regression Foundations",
         "Understand what regression is really optimising.",
         ["Train/test split", "DummyRegressor baseline", "Linear regression / OLS",
          "Loss functions: MSE, RMSE, MAE", "Residuals", "Bias vs variance",
          "Overfitting vs underfitting"],
         ["Target histogram", "Feature vs target scatter",
          "Actual vs predicted", "Residuals vs predicted", "Residual histogram"],
         "Are my errors random, or is the model missing structure?",
         "California housing or diabetes regression"),

        ("Phase 2", "Preprocessing & Data Quality",
         "Data preparation often matters more than model choice.",
         ["Missing values: MCAR, MAR, MNAR", "Simple / KNN / MICE imputation",
          "Scaling: StandardScaler, RobustScaler", "Outlier handling",
          "Categorical encoding", "Data leakage"],
         ["Missingness bar plot", "Missingness heatmap",
          "Boxplots for outliers", "Target by category boxplots"],
         "Is this missing value actually missing, or does it mean something?",
         "Ames Housing ID 42165"),

        ("Phase 3", "Regularisation & Feature Selection",
         "Understand how models become stable.",
         ["Ridge, Lasso, ElasticNet", "Coefficient shrinkage",
          "Multicollinearity / VIF", "Coefficient paths"],
         ["Coefficient path plot", "CV error vs alpha",
          "Correlation heatmap", "Feature importance plot"],
         "Which features are useful, redundant, or dangerous?",
         "Diabetes sklearn / CPU Activity ID 197"),

        ("Phase 4", "Nonlinearity & Interactions",
         "Learn when linear models are too simple.",
         ["Polynomial features", "Interaction terms", "Splines",
          "RBF/kernel features", "Log transforms", "Mutual information",
          "Partial dependence plots"],
         ["Scatter + smooth curve", "Feature vs target colored by another feature",
          "Partial dependence plots", "Residuals before/after nonlinear features"],
         "Is the relationship additive and linear, or curved and interactive?",
         "CPU activity ID 197, Ames housing, Insurance charges"),

        ("Phase 5", "Tree Models & Ensembles",
         "Learn the strongest tabular regression models.",
         ["Decision trees", "Random Forest, ExtraTrees",
          "Gradient Boosting, XGBoost / LightGBM / CatBoost",
          "Feature importance", "Permutation importance"],
         ["Model comparison bar chart", "Feature importance",
          "Permutation importance", "Residuals by feature"],
         "Does a nonlinear model discover interactions better than manual features?",
         "Bike Sharing ID 42712, Ames housing, California housing"),

        ("Phase 6", "Metrics & Target Distributions",
         "Choose the right metric for the target.",
         ["RMSE, MAE, RMSLE, MAPE", "Median absolute error",
          "Poisson / Tweedie deviance", "Pinball / Quantile loss",
          "Rule: normal target &#8594; RMSE, skewed price &#8594; RMSLE, "
          "count &#8594; Poisson/Tweedie"],
         ["Metric comparison bar chart", "Residual distribution"],
         "Does my metric match the real cost of being wrong?",
         "Housing prices, Bike counts, Insurance claims, Wine quality"),

        ("Phase 7", "Specialised Regression",
         "Learn that regression is not one single problem.",
         ["Count regression: Poisson, Negative Binomial, Tweedie",
          "Quantile regression", "Ordinal regression",
          "Multi-output regression", "Robust regression",
          "Spatial regression", "Time-aware regression"],
         ["Quantile prediction bands", "Calibration plots",
          "Spatial residual maps"],
         "What type of target am I predicting?",
         "Abalone, Wine Quality 287, Energy Efficiency 242, California 537"),

        ("Phase 8", "Validation Strategy",
         "Learn how not to fool yourself.",
         ["KFold, Repeated KFold, GroupKFold",
          "TimeSeriesSplit, Spatial CV, Nested CV",
          "Rule: random rows &#8594; KFold, time data &#8594; TimeSeriesSplit, "
          "repeated groups &#8594; GroupKFold, geographic &#8594; spatial split"],
         ["CV score distribution boxplot", "Train vs CV gap plot"],
         "Does my validation setup match future deployment?",
         "Any dataset with a temporal or group structure"),

        ("Phase 9", "Error Analysis",
         "Stop asking only 'what is the score?' — ask 'where does it fail?'",
         ["Residual analysis", "Segment-level errors",
          "Error by category / target range",
          "Largest-error inspection", "Heteroscedasticity"],
         ["Residuals vs prediction", "Top error table",
          "Error distribution", "Residuals by category"],
         "Which cases does the model systematically misunderstand?",
         "Ames Housing — analyse by Neighborhood, BldgType"),

        ("Phase 10", "Uncertainty & Production",
         "Make regression useful in real systems.",
         ["Prediction intervals", "Quantile regression intervals",
          "Conformal prediction", "Drift monitoring",
          "Missing-rate monitoring", "Model versioning",
          "Retraining triggers"],
         ["Coverage plot (actual vs PI)", "Drift monitoring line charts"],
         "How confident is the model, and is the incoming data still familiar?",
         "Any dataset — build train / predict / monitor pipeline"),
    ]

    for tag, title, goal, concepts, plots, key_q, dataset in phases:
        block = []
        block.append(Paragraph(f"{tag}: {title}", st["h2"]))
        block.append(Paragraph(f"<i>Goal:</i> {goal}", st["body"]))
        block.append(Spacer(1, 4))

        rows = []
        for i, (concept, plot) in enumerate(
                zip(concepts, plots + [""]*(max(0,len(concepts)-len(plots))))):
            rows.append([
                Paragraph(f"&#x2022; {concept}",
                           ParagraphStyle("td2", fontSize=9, leading=13,
                                          fontName="Helvetica")),
                Paragraph(f"&#x25C6; {plot}" if plot else "",
                           ParagraphStyle("td2", fontSize=9, leading=13,
                                          fontName="Helvetica",
                                          textColor=COVERS[S])),
            ])
        # pad if plots longer
        for plot in plots[len(concepts):]:
            rows.append([Paragraph("", ParagraphStyle("td2", fontSize=9)),
                         Paragraph(f"&#x25C6; {plot}",
                                    ParagraphStyle("td2", fontSize=9, leading=13,
                                                   fontName="Helvetica",
                                                   textColor=COVERS[S]))])
        t = Table(rows, colWidths=[(W-2*MARGIN)*0.55, (W-2*MARGIN)*0.45])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), LIGHT[S]),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("LINEAFTER",     (0,0), (0,-1),  1, RULE_CLR[S]),
            ("ROWBACKGROUNDS",(0,0), (-1,-1), [LIGHT[S], colors.white]),
        ]))
        block.append(t)
        block.append(Spacer(1, 4))

        q_para = Paragraph(f'<i>Key question: "{key_q}"</i>', st["q"])
        block.append(q_para)
        ds_para = Paragraph(f"<b>Dataset:</b> {dataset}",
                             ParagraphStyle("ds", fontSize=9, leading=13,
                                            textColor=COVERS[S],
                                            fontName="Helvetica-Oblique"))
        block.append(ds_para)
        block.append(Spacer(1, 8))
        story.append(KeepTogether(block))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3  ─  OpenML Dataset Map
# ═══════════════════════════════════════════════════════════════════════════════
def build_section3(story):
    S = 2
    st = make_styles(S)
    story.append(PageBreak())
    section_cover(story, S, "", "OpenML Dataset Recommendations",
                  "The right dataset for each regression learning phase")
    story.append(Spacer(1, 10))

    story.append(Paragraph("Full Dataset Map", st["h1"]))
    rows = [
        ("Regression basics",       "Diabetes",            "44214 / sklearn",  "Small, clean — OLS, RMSE, MAE, residuals"),
        ("Full practical regression","Ames Housing",        "42165",            "Missing data, categorical, outliers, skewed target"),
        ("Alternative housing",     "Ames Housing cleaned", "43926",            "Similar concept, cleaner version"),
        ("Nonlinearity / basis fn", "CPU activity",        "197",              "Polynomial features, splines, kernels"),
        ("Regularisation",          "Diabetes / CPU",      "44214, 197",       "Ridge, Lasso, ElasticNet, coefficient paths"),
        ("Categorical encoding",    "Medical charges",     "42720",            "Categorical + numeric, skewed cost target"),
        ("Missing data & robust",   "Ames Housing",        "42165",            "MICE, KNN, MNAR/MAR, Huber regression"),
        ("Gradient boosting",       "Bike Sharing",        "42712",            "Time features, boosting, RMSLE"),
        ("Spatial regression",      "California Housing",  "537",              "Latitude/longitude, spatial leakage, spatial CV"),
        ("Heavy-tailed target",     "Diamonds / medical",  "42225, 42720",     "Log target, Box-Cox, Yeo-Johnson"),
        ("Count regression",        "Bike Sharing/Abalone","42712, 1",         "Count-like targets, Poisson/Tweedie"),
        ("Quantile regression",     "Wine quality",        "287 or 40691",     "Quantile loss, prediction intervals"),
        ("Ordinal/rating target",   "Wine quality/student","287, 40536",       "Ordered target, ordinal vs regression"),
        ("Multi-output regression", "Energy efficiency",   "242",              "Heating + cooling load joint prediction"),
        ("Large tabular / NNs",     "HIGGS",               "23512",            "Large-scale tabular, MLP/TabNet"),
        ("Production / drift",      "Bike / House sales",  "42712, 42165",     "Time drift, monitoring, train profile"),
    ]
    headers = ["Learning Phase", "Dataset", "OpenML ID", "Why It Is Useful"]
    cw = [(W-2*MARGIN)*0.24, (W-2*MARGIN)*0.22,
          (W-2*MARGIN)*0.12, (W-2*MARGIN)*0.42]
    tdata = [[Paragraph(f"<b>{h}</b>",
                        ParagraphStyle("th", fontSize=9, leading=13,
                                       textColor=colors.white,
                                       fontName="Helvetica-Bold")) for h in headers]]
    for r in rows:
        tdata.append([Paragraph(c,
                                ParagraphStyle("td", fontSize=9, leading=13,
                                               fontName="Helvetica",
                                               textColor=colors.HexColor("#222222")))
                      for c in r])
    t = Table(tdata, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  COVERS[S]),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [LIGHT[S], colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.4, RULE_CLR[S]),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 7),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Top 8
    story.append(Paragraph("Top 8 Essential Datasets", st["h1"]))
    top8 = [
        ("1", "Ames Housing",     "42165", "Missing data, encoding, outliers, skewed target, full workflow"),
        ("2", "Diabetes",         "sklearn","Small, clean — perfect for OLS, residuals, regularisation"),
        ("3", "CPU Activity",     "197",   "Nonlinear features, polynomial, splines, kernels"),
        ("4", "Bike Sharing",     "42712", "Time features, gradient boosting, RMSLE"),
        ("5", "California Housing","537",  "Spatial leakage, lat/lon features, spatial CV"),
        ("6", "Wine Quality",     "287",   "Quantile/ordinal targets, prediction intervals"),
        ("7", "Energy Efficiency","242",   "Multi-output regression, joint targets"),
        ("8", "Diamonds",         "42225", "Heavy-tailed/log-normal target, Box-Cox"),
    ]
    for num, ds, dsid, why in top8:
        row_data = [[
            Paragraph(f"<b>{num}</b>",
                       ParagraphStyle("n", fontSize=14, leading=18,
                                      textColor=COVERS[S],
                                      fontName="Helvetica-Bold",
                                      alignment=TA_CENTER)),
            Paragraph(f"<b>{ds}</b><br/>"
                      f'<font color="#888888" size="9">ID {dsid}</font>',
                       ParagraphStyle("ds", fontSize=11, leading=16,
                                      fontName="Helvetica-Bold")),
            Paragraph(why, ParagraphStyle("wh", fontSize=9, leading=14,
                                           fontName="Helvetica",
                                           textColor=colors.HexColor("#333333"))),
        ]]
        t2 = Table(row_data, colWidths=[22, 110, W-2*MARGIN-140])
        t2.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), LIGHT[S]),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("LINEAFTER",     (0,0), (0,-1),  2, COVERS[S]),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(t2)
        story.append(Spacer(1, 4))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4  ─  Industry Standards
# ═══════════════════════════════════════════════════════════════════════════════
def build_section4(story):
    S = 3
    st = make_styles(S)
    story.append(PageBreak())
    section_cover(story, S, "", "Industry ML Engineering Standards",
                  "15 concepts that separate production ML from notebook ML")
    story.append(Spacer(1, 10))

    standards = [
        ("1  Data Contracts & Schema Validation",
         ["Required columns, data types, allowed categories",
          "Value ranges, missing-rate thresholds, duplicate checks",
          "Tools: pandera, pydantic, Great Expectations"],
         "Will this pipeline fail safely if tomorrow's data is different?"),
        ("2  Leakage Prevention",
         ["Target leakage, time leakage, group leakage",
          "Preprocessing leakage — impute AFTER split",
          "Feature availability at prediction time — does it exist then?"],
         "Would this feature actually exist at prediction time?"),
        ("3  Validation Strategy Selection",
         ["KFold for random rows, TimeSeriesSplit for time data",
          "GroupKFold when same user/group appears many times",
          "Spatial split for geographic data, Nested CV for small datasets"],
         "Does validation match deployment?"),
        ("4  Metric Design",
         ["RMSE, MAE, RMSLE, MAPE, sMAPE, Median absolute error",
          "Quantile / pinball loss, business-weighted error",
          "Segment-level metrics — overall RMSE can hide group failures"],
         "What kind of mistake is expensive?"),
        ("5  Baselines Before Fancy Models",
         ["Mean baseline, median baseline, last-value for time data",
          "Group-average baseline, simple linear model",
          "Previous production model — always beat it first"],
         "Did the model beat a simple, explainable baseline?"),
        ("6  Reproducible Pipelines",
         ["Pipeline + ColumnTransformer — same code for train and predict",
          "Saved preprocessors, fixed random seeds, config files",
          "Versioned artifacts: model.joblib, metrics.json, training_profile.json"],
         "Can someone retrain this model six months later?"),
        ("7  Experiment Tracking",
         ["Track parameters, metrics, dataset version, model version",
          "Track feature set and all plots/artifacts",
          "Tools: MLflow, Weights & Biases, DVC, Neptune"],
         "Which experiment produced this model, and why was it chosen?"),
        ("8  Hyperparameter Tuning",
         ["Grid search &#8594; random search &#8594; Bayesian (Optuna)",
          "Always tune after strong baseline — not before",
          "Nested CV for honest tuning to avoid validation overfitting"],
         "Did tuning improve generalisation or just overfit validation?"),
        ("9  Model Interpretability",
         ["Linear coefficients, feature importance, permutation importance",
          "Partial dependence plots (PDP), ICE plots",
          "SHAP values — global bar/beeswarm, local waterfall"],
         "Can we explain why the prediction changed?"),
        ("10  Error Analysis By Segment",
         ["Error by category, geography, price band, customer segment",
          "Error by time period, error by missingness pattern",
          "Plots: residuals by segment, MAE by category, top 50 errors"],
         "Who or what does the model fail on?"),
        ("11  Uncertainty Estimation",
         ["Prediction intervals, quantile regression, conformal prediction",
          "Bootstrap intervals, coverage check, interval width",
          "Calibration: does predicted 80% PI actually contain 80% of actuals?"],
         "Should we trust this prediction, and how uncertain is it?"),
        ("12  Robustness Testing",
         ["Missing column test, new category test, extreme value test",
          "Outlier stress test, distribution shift test",
          "Data corruption test — test what happens with messy production data"],
         "What happens when production data is messy?"),
        ("13  Monitoring After Deployment",
         ["Input drift, prediction drift, missing-rate drift, category drift",
          "Performance drift — MAE/RMSE when actuals arrive",
          "Retraining triggers: KS test p-value, missing rate threshold"],
         "Is the model still valid today?"),
        ("14  Model Governance",
         ["Model card: intended use, limitations, assumptions, out-of-scope",
          "Dataset card: collection method, biases, version",
          "Audit trail, approval workflow — required in regulated industries"],
         "Can this model be reviewed, audited, and trusted?"),
        ("15  Deployment Basics",
         ["Batch prediction vs real-time API prediction",
          "Scheduled retraining, feature store basics",
          "Docker basics, CI/CD for ML, Model registry"],
         "How does this model actually reach users?"),
    ]

    for title, bullets, key_q in standards:
        block = []
        block.append(Paragraph(title, st["h2"]))
        info_box(story, bullets, S)
        story.append(Paragraph(f'<i>Industry question: "{key_q}"</i>', st["q"]))
        story.append(Spacer(1, 8))

    story.append(Spacer(1, 6))
    hr(story, S)
    story.append(Paragraph("Ideal Artifacts Per Model", st["h1"]))
    art = [
        "model.joblib — serialised fitted pipeline",
        "metrics.json — all evaluation metrics",
        "cv_results.csv — all cross-validation fold results",
        "training_profile.json — quantiles for drift monitoring",
        "schema.json / pandera schema — data contract",
        "feature_importance.csv — ranked features",
        "predictions.csv — test set predictions",
        "error_analysis.csv — largest errors with features",
        "model_card.md — intended use and limitations",
        "plots/ — target, missingness, actual_vs_predicted, residuals, comparison",
    ]
    info_box(story, art, S, "Deliver this folder for every pipeline:")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5  ─  Classification Learning Plan
# ═══════════════════════════════════════════════════════════════════════════════
def build_section5(story):
    S = 4
    st = make_styles(S)
    story.append(PageBreak())
    section_cover(story, S, "", "Classification Learning Plan",
                  "11-phase path from binary foundations to multi-label and production drift")
    story.append(Spacer(1, 10))

    story.append(Paragraph("Core Mental Model", st["h1"]))
    story.append(Paragraph(
        "Classification is not only: logistic &#8594; SVM &#8594; trees &#8594; boosting.<br/>"
        "It is really: <b>predict probability &#8594; choose threshold &#8594; "
        "measure cost &#8594; explain errors &#8594; monitor drift</b>",
        st["body"]))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Learning Loop", st["h2"]))
    loop_items = [
        "Start with DummyClassifier",
        "Build simple logistic regression",
        "Compare probability scores",
        "Choose decision threshold",
        "Analyse confusion matrix",
        "Compare algorithms",
        "Calibrate probabilities",
        "Analyse errors by segment",
        "Save model + metrics + profile",
        "Monitor data and performance drift",
    ]
    info_box(story, loop_items, S)

    phases = [
        ("Phase 1", "Binary Classification Foundations",
         "Understand what classification is really doing.",
         ["Binary labels, class balance, DummyClassifier",
          "Logistic regression, sigmoid, log-odds, log loss",
          "Decision boundary, regularisation, confusion matrix"],
         "Titanic, Heart disease, Breast cancer",
         "Can my model beat a simple baseline, and what kind of mistakes does it make?"),
        ("Phase 2", "Metrics & Thresholds",
         "One of the most important classification phases.",
         ["Accuracy, Precision, Recall, F1, F-beta, Balanced accuracy",
          "ROC AUC, PR AUC, MCC, Log loss, Brier score",
          "Threshold tuning — 0.5 is almost never the optimal threshold",
          "Rule: imbalanced &#8594; PR AUC; FN costly &#8594; recall; FP costly &#8594; precision"],
         "German credit, Bank marketing, Credit card fraud",
         "Is 0.5 really the right threshold? (Usually, no.)"),
        ("Phase 3", "Data Preparation for Classification",
         "Learn preprocessing that respects classification structure.",
         ["Missing data, categorical encoding, ordinal encoding",
          "Rare category handling, class imbalance, leakage prevention",
          "class_weight='balanced', SMOTE — applied only inside CV folds"],
         "Titanic, Adult income, German credit, Hepatitis",
         "Am I preprocessing in a way that leaks validation information into training?"),
        ("Phase 4", "Linear Classifiers",
         "Build intuition for linear decision boundaries.",
         ["Logistic regression, regularised logistic regression",
          "Linear SVM, SGDClassifier, Naive Bayes",
          "Coefficients as explanations — log-odds interpretation"],
         "Breast cancer, Spambase, SMS spam",
         "Is the class boundary mostly linear?"),
        ("Phase 5", "Nonlinear Boundaries",
         "Learn when linear classifiers are insufficient.",
         ["Kernel SVM (RBF, polynomial), Decision trees, k-NN",
          "Decision boundary visualisation",
          "Overfitting with flexible models — tree depth vs train/test score"],
         "Breast cancer, Sonar, Two-moons synthetic",
         "Do I need nonlinear boundaries, or am I just overfitting?"),
        ("Phase 6", "Trees & Ensembles",
         "Master the strongest tabular classification models.",
         ["Decision trees, Random Forest, ExtraTrees",
          "Gradient Boosting, XGBoost / LightGBM / CatBoost",
          "Feature importance, Permutation importance, SHAP"],
         "Adult income, Bank marketing, Covertype, Credit-g",
         "Does the ensemble improve generalisation or only training performance?"),
        ("Phase 7", "Class Imbalance",
         "Deserves its own phase — the most common production problem.",
         ["Why accuracy fails on imbalanced data",
          "class_weight='balanced', SMOTE, ADASYN, Tomek links",
          "Balanced Random Forest, threshold tuning",
          "Evaluate with PR AUC, MCC — not accuracy"],
         "Credit card fraud, Mammography, Bank marketing",
         "Did I improve minority class detection, or only move errors around?"),
        ("Phase 8", "Probability Calibration",
         "Very industry-relevant — probabilities must be honest.",
         ["Calibration vs discrimination",
          "Reliability diagram, Brier score, Expected Calibration Error",
          "Platt scaling, isotonic calibration, CalibratedClassifierCV"],
         "Breast cancer, German credit, Bank marketing",
         "When the model says 80%, is it correct about 80% of the time?"),
        ("Phase 9", "Multiclass Classification",
         "Extend binary classification to K classes.",
         ["Softmax, One-vs-Rest, One-vs-One",
          "Macro / micro / weighted F1, Top-k accuracy",
          "Multiclass confusion matrix, per-class ROC AUC"],
         "MNIST, Fashion-MNIST, Covertype, Wine",
         "Which classes are confused with each other, and why?"),
        ("Phase 10", "Feature Selection",
         "Find features that are genuinely predictive.",
         ["Mutual information, Chi-square, ANOVA F-test",
          "L1 logistic regression, RFE, RFECV, SelectFromModel",
          "Feature importance stability across bootstrap samples"],
         "Madelon, Gisette, Spambase",
         "Which features are genuinely predictive and stable across folds?"),
        ("Phase 11", "Multi-label & Ordinal Classification",
         "Special classification structures beyond single labels.",
         ["Multi-label: binary relevance, classifier chains, label powerset",
          "Evaluation: Hamming loss, Jaccard score, subset accuracy",
          "Ordinal: Frank-Hall decomposition, Quadratic Weighted Kappa (QWK)",
          "Datasets: Scene/Yeast (multi-label); Wine quality/ESL (ordinal)"],
         "Scene ID 312, Yeast, Wine quality 287",
         "Does each row have one class, many labels, or an ordered class?"),
    ]

    for tag, title, goal, concepts, dataset, key_q in phases:
        block = []
        block.append(Paragraph(f"{tag}: {title}", st["h2"]))
        block.append(Paragraph(f"<i>Goal:</i> {goal}", st["body"]))
        block.append(Spacer(1, 3))
        for c in concepts:
            block.append(Paragraph(f"&#x2022; {c}", st["bullet"]))
        block.append(Spacer(1, 3))
        block.append(Paragraph(
            f'<font color="{COVERS[S].hexval() if hasattr(COVERS[S],"hexval") else "#7F77DD"}"><b>Datasets:</b></font> {dataset}',
            ParagraphStyle("ds5", fontSize=9, leading=13, fontName="Helvetica",
                           textColor=COVERS[S])))
        block.append(Paragraph(f'<i>Key question: "{key_q}"</i>', st["q"]))
        block.append(Spacer(1, 8))
        story.append(KeepTogether(block))

# ── Page number footer ────────────────────────────────────────────────────────
_sec_colors = COVERS

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#888888"))
        self.drawRightString(A4[0] - MARGIN,
                             12*mm,
                             f"Page {self._pageNumber} of {page_count}  |  ML Practice Notebook")

# ── Build PDF ─────────────────────────────────────────────────────────────────
def build_pdf():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=16*mm, bottomMargin=22*mm,
        title="ML Practice Notebook",
        author="ML Learning Notes",
    )
    story = []
    build_section1(story)
    build_section2(story)
    build_section3(story)
    build_section4(story)
    build_section5(story)
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF saved: {OUT}")

build_pdf()
