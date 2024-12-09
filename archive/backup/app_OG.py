import dash
from dash import Dash, dcc, html, dash_table, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import io
import base64
import seaborn as sns
import plotly.figure_factory as ff
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import plotly.io as pio

plotly_template = pio.templates["ggplot2"]

pio.templates.default = plotly_template


def generate_sample_csv():
    """
    Generate a sample CSV with the correct columns
    
    Returns:
        pandas.DataFrame: Sample CSV dataframe
    """
    # Create a minimal sample dataframe with all required columns
    sample_data = pd.DataFrame({
        'employee_id': [1, 2, 3],
        'university': ['Stanford University', 'MIT', 'Harvard University'],
        'graduation_grade': [3.8, 3.9, 3.7],
        'department': ['Engineering', 'Data Science', 'Marketing'],
        'job_title': ['Software Engineer', 'Data Analyst', 'Marketing Specialist'],
        'start_date': ['2023-01-15', '2023-02-20', '2023-03-10'],
        'performance_score': [85.5, 88.2, 82.7],
        'years_of_experience': [2.5, 1.8, 3.0],
        'age': [28, 26, 30],
        'salary': [95000, 85000, 90000]
    })
    
    return sample_data

def create_ml_insights(df):
    """
    Generate machine learning insights for predicting performance score
    
    Args:
        df (pandas.DataFrame): Input dataframe
    
    Returns:
        dict: Machine learning insights for performance prediction
    """
    # Prepare features for ML
    features = ['graduation_grade', 'years_of_experience', 'age', 'salary']
    X = df[features]
    y = df['performance_score']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Performance Prediction Model
    # Split data (simple train/test split)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Linear Regression Model - use other model
    from sklearn.linear_model import LinearRegression
    performance_predictor = LinearRegression()
    performance_predictor.fit(X_train, y_train)
    
    # Model Performance
    train_score = performance_predictor.score(X_train, y_train)
    test_score = performance_predictor.score(X_test, y_test)

    # 2. Clustering
    # Use K-Means clustering for grouping employees
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Dimensionality Reduction for Visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Prepare clustering visualization data
    clustering_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': cluster_labels,
        'Performance': y,  # Use performance score in clustering data
        'Department': df['department']
    })

    # 3. Feature Importance for Performance Prediction
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(performance_predictor.coef_)  # Absolute value of coefficients for importance
    }).sort_values('Importance', ascending=False)

    return {
        'performance_predictor': performance_predictor,
        'scaler': scaler,
        'train_score': train_score,
        'test_score': test_score,
        'clustering_df': clustering_df,
        'feature_importance': feature_importance
    }


def create_dash_app():
    """
    Create Dash application with upload and download functionality
    
    Returns:
        dash.Dash: Configured Dash application
    """
        
    app = Dash(__name__, external_stylesheets=[
        'https://codepen.io/chriddyp/pen/bWLwgP.css',  # Basic Dash CSS
        '/assets/custom.css'  # Custom CSS file
    ])
    
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content', style={'position': 'absolute', 'width': '100%', 'height': '100vh', 'zIndex': -1}),
        html.Button('← Back to Upload', id='return-upload-btn',
                className='return-btn',
                style={'display': 'none'}),
        
        html.Div([
            html.H1('Employee Data Dashboard', className='main-title'),
            
            # Instructions paragraph

            # File Upload and Download Section
            html.Div([
                html.H3('How to use this tool:'),
                html.P(
                    """
                    Welcome to the Employee Data Dashboard! It will help you draw insights about your workfroce
                    and their performance.
                    You can upload your own CSV file or use the demo data
                    to explore dashboard functionality.
                    """,
                    className='instructions'
                ),
                html.Div([
                    html.Button('Run Demo', id='run-demo-btn', n_clicks=0, className='btn-primary')
                ], className='upload-section'),

                # html.Hr(className="upload_divider"),

                html.P(
                    """
                    If you want to use your own data, upload your csv file below.
                    Please make sure your csv file matches the columns of the template csv you can also download below.
                    """,
                    className='instructions'
                ),
                
                # CSV File Upload
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ], className='upload-text'),
                        multiple=False
                    ),
                ], className='upload-section'),

                # Download Sample CSV Button
                html.Div([
                    html.Button('Download Template CSV', id='download-sample-btn', n_clicks=0, className='btn-secondary'),
                    dcc.Download(id='download-sample-csv')
                ], className='upload-section'),
                # html.Hr(className="upload_divider"),
                html.Div([
                    html.P(
                        """
                        © 2024 Sijana Mamos
                        """
                    ),
                    html.A(
                        href='https://github.com/Sijana',
                        target='_blank',
                        children=html.Img(src='/assets/icons8-github-100.png',
                                          style={'width': '40px', 'margin': '10px'})
                        ),
                    html.A(
                        href='https://www.linkedin.com/in/sijana-mamos',
                        target='_blank',
                        children=html.Img(src='/assets/LI.png',
                                          style={'width': '40px', 'margin': '10px'})
                        )
                    ], style={'textAlign': 'center', 'marginTop': '20px'}),
            ], className='upload-container', id='upload-section'),
            
            # Validation Message
            html.Div(id='upload-validation', className='validation-message'),
            
            # Filters and Dashboard (Initially Hidden)
            html.Div(id='dashboard-content', style = {
                                                    'display': 'none',
                                                    'backgroundColor': '#570225',
                                                    'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)',  # Subtle shadow for depth
                                                    'padding': '20px',        # Space inside the container
                                                    'margin': '0 auto',         # Space outside the container
                                                    'width': '90%',           # Responsive width
                                                    'maxWidth': '90%',     # Limit the maximum width
                                                    'minHeight': '50%',     # Set a minimum height to ensure consistent size
                                                    'color': '#333',          # Dark text color for readability
                                                    'textAlign': 'center',    # Center align text within the container
                                                    'overflow': 'auto',       # Allow scrolling if content overflows
                                                }, children=[
                html.H2("Filters and General Stats"),
                html.Div([
                    html.Div([
                        html.Label('Filter By University'),
                        dcc.Dropdown(
                            id='university-dropdown',
                            placeholder='All Universities',
                            className='custom-dropdown'
                        )
                    ], className='filter-section'),

                    html.Div([
                        html.Label('Filter By Department'),
                        dcc.Dropdown(
                            id='department-dropdown',
                            placeholder='All Departments',
                            className='custom-dropdown'
                        )
                    ], className='filter-section'),

                    # Additional Filter for Job Title
                    html.Div([
                        html.Label('Filter By Job Title'),
                        dcc.Dropdown(
                            id='job-title-dropdown',
                            placeholder='All Job Titles',
                            className='custom-dropdown'
                        )
                    ], className='filter-section'),

                ], className='filter-container'),

                # Hidden Demo State Container
                html.Div(id='run-demo-state', style={'display': 'none'}),
                # Insights Container
                html.Div(id='insights-container', className='insights-container'),
                html.H2("Performance Visualizations"),
                # Performance Visualizations
                html.Div([

                    # New visualizations
                    html.Div([
                        # html.P("A Bar Chart showing Performance Score vs. University."),
                        dcc.Graph(id='performance-by-university')
                    ], className='graph-container'),
                    html.Div([
                        dcc.Graph(id='performance-heatmap')
                    ], className='graph-container')
                ], className='visualization-container'),
                html.H2("Salary Visualizations"),
                # Salary Visualizations
                html.Div([
                    html.Div([
                        dcc.Graph(id='salary-distribution')
                    ], className='graph-container'),
                    html.Div([
                        dcc.Graph(id='salary-department-boxplot')
                    ], className='graph-container'),
                    html.Div([
                        dcc.Graph(id='experience-salary-scatter')
                    ], className='graph-container'),
                ], className='visualization-container'),

                #ML Insights

                html.Div([
                html.H2('Machine Learning Insights'),

                # Performance Prediction Section
                html.Div([
                    html.H3('Performance Prediction Model'),
                    html.Div(id='performance-prediction-model-insights'),
                    html.P("""
                    This section shows how well we can predict performance score based on different employee characteristics.
                    The accuracy percentages tell you how reliable our prediction model is.
                    A higher percentage means the model is better at guessing perfornaces.
                    """),
                    html.Div([
                        html.Label('Graduation Grade (3.0-4.0): '),
                        dcc.Input(id='demo-grade', type='number', min=3.0, max=4.0, step=0.1, value=3.5),
                        html.Label('Years of Experience (0-10): '),
                        dcc.Input(id='demo-experience', type='number', min=0, max=10, step=0.5, value=3),
                        html.Button('Predict Performance', id='predict-performance-btn', className='btn-secondary')
                        ]),
                        html.Div(id='performance-prediction-result')
                    ], className='ml-section'),
                
                    # Feature Importance Section
                    html.Div([
                        html.H3('Feature Importance for Performance'),
                        dcc.Graph(id='feature-importance-chart'),
                        html.P("""
                        This bar chart shows which factors have the biggest impact on an employee's performance.
                        Taller bars mean that particular characteristic (like years of experience or grades)
                        has a stronger influence on determining performance. Think of it like a 'performance influence meter'.
                        """)
                    ], className='ml-section'),

                    # Clustering Visualization
                    html.Div([
                        html.H3('Employee Clustering'),
                        dcc.Graph(id='employee-clustering-chart'),
                        html.P("""
                        This scatter plot groups employees with similar characteristics into clusters.
                        Each dot represents an employee, and colors show different groups.
                        Employees close to each other on the chart are more similar in terms of
                        their work experience, performance, and other key attributes.
                        It's like sorting employees into teams based on their shared traits.
                        """)
                    ], className='ml-section')
                ], className='ml-insights-container'),

                # Data Table
                html.Div([
                    html.H3('Filtered Employee Data', style={'color': '#F1F1F1'}),  # White color for header text
                    dash_table.DataTable(
                        id='data-table',
                        page_size=10,
                        style_table={
                            'overflowX': 'auto',
                            'backgroundColor': '#2B2B2B',  # Dark background for the table (ggplot2 dark gray)
                            'border': '1px solid #444',  # Border color for the table, subtle but noticeable
                            'color': '#F1F1F1'  # Light gray # Set text color to white
                        },
                        style_cell={
                            'textAlign': 'left',
                            'backgroundColor': 'black',  # Set cell background to black
                            'color': '#F1F1F1',  # Light gray text for readability
                            'border': '1px solid #444',  # Subtle border for each cell
                            'fontFamily': 'Arial, sans-serif',  # Font for consistencyal: set font for the table
                        },
                        style_header={
                            'backgroundColor': '#4C4C4C',  # Slightly lighter dark gray for the header row
                            'color': '#FFFFFF',  # White text in the header for contrast
                            'fontWeight': 'bold',  # Bold header text for emphasis
                            'textAlign': 'center'
                        }
                    )
                ], className='table-container')
            ])
        ], className='main-container')
    ])
    
    server = app.server

    # Callback for downloading sample CSV
    @app.callback(
        Output('download-sample-csv', 'data'),
        Input('download-sample-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_sample_csv(n_clicks):
        sample_df = generate_sample_csv()
        return dcc.send_data_frame(sample_df.to_csv, 'sample_hiring_data.csv', index=False)

    @app.callback(
        dash.dependencies.Output('page-content', 'children'),
        [dash.dependencies.Input('url', 'pathname')]
    )
    def display_page(pathname):
        if pathname == '/':
            return html.Iframe(src="/assets/particles.html", width="100%", height="100%")
        return '404, Not found'

    @app.callback(
        [
            Output('upload-validation', 'children'),
            Output('dashboard-content', 'style'),
            Output('page-content', 'style'),
            Output('university-dropdown', 'options'),
            Output('department-dropdown', 'options'),
            Output('job-title-dropdown', 'options'),
            Output('university-dropdown', 'value'),
            Output('department-dropdown', 'value'),
            Output('run-demo-state', 'children'),
            Output('upload-section', 'style'),
            Output('return-upload-btn', 'style'),
        ],
        [
            Input('upload-data', 'contents'),
            Input('run-demo-btn', 'n_clicks'),
            Input('return-upload-btn', 'n_clicks'),
        ],
        [State('upload-data', 'filename'), State('page-content', 'style'), State('dashboard-content', 'style')],
        prevent_initial_call=True,
    )
    def handle_app_flow(contents, demo_clicks, return_clicks, filename, page_content_style, dashboard_content_style):
        ctx = dash.callback_context

        if not ctx.triggered:
            return (
                '',
                {**dashboard_content_style, 'display': 'none'},  # Preserve other styles, change display
                {**page_content_style, 'height': '100vh', 'display': 'block'},  # Preserve other styles, change display
                [],  # Empty dropdowns
                [],
                [],
                None,
                None,
                None,
                {'display': 'block'},  # Show upload section
                {'display': 'none'},  # Hide return button
            )

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Handle return to upload
        if triggered_id == 'return-upload-btn':
            return (
                '',
                {**dashboard_content_style, 'display': 'none'},  # Preserve other styles, change display
                {**page_content_style, 'height': '100vh', 'display': 'block'},  # Preserve other styles, change display
                [],  # Reset dropdowns
                [],
                [],
                None,
                None,
                None,
                {'display': 'block'},  # Show upload section
                {'display': 'none'},  # Hide return button
            )

        # Initialize variables
        run_demo_state = None
        df = None

        # Handle "Run Demo" click
        if triggered_id == 'run-demo-btn':
            demo_csv_path = 'demo_data.csv'
            try:
                df = pd.read_csv(demo_csv_path)
                run_demo_state = 'demo'
            except Exception as e:
                return (
                    f"Error loading demo data: {str(e)}",
                    {**dashboard_content_style, 'display': 'none'},  # Preserve other styles, change display
                    {**page_content_style, 'height': '100vh', 'display': 'block'},  # Preserve other styles, change display
                    [],  # Reset dropdowns
                    [],
                    [],
                    None,
                    None,
                    None,
                    {'display': 'block'},  # Show upload section
                    {'display': 'none'},  # Hide return button
                )

        # Handle file upload
        elif triggered_id == 'upload-data':
            if not contents:
                return (
                    "No file uploaded.",
                    {**dashboard_content_style, 'display': 'none'},  # Preserve other styles, change display
                    {**page_content_style, 'height': '100vh', 'display': 'block'},  # Preserve other styles, change display
                    [],  # Reset dropdowns
                    [],
                    [],
                    None,
                    None,
                    None,
                    {'display': 'block'},  # Show upload section
                    {'display': 'none'},  # Hide return button
                )

            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            except Exception as e:
                return (
                    f"Error processing file: {str(e)}",
                    {**dashboard_content_style, 'display': 'none'},  # Preserve other styles, change display
                    {**page_content_style, 'height': '100vh', 'display': 'block'},  # Preserve other styles, change display
                    [],  # Reset dropdowns
                    [],
                    [],
                    None,
                    None,
                    None,
                    {'display': 'block'},  # Show upload section
                    {'display': 'none'},  # Hide return button
                )

        # Validate data
        required_columns = [
            'employee_id', 'university', 'graduation_grade',
            'department', 'job_title', 'start_date',
            'performance_score', 'years_of_experience',
            'age', 'salary'
        ]
        if df is not None:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return (
                    f"Missing columns: {', '.join(missing_columns)}",
                    {**dashboard_content_style, 'display': 'none'},  # Preserve other styles, change display
                    {**page_content_style, 'height': '100vh', 'display': 'block'},  # Preserve other styles, change display
                    [],  # Reset dropdowns
                    [],
                    [],
                    None,
                    None,
                    None,
                    {'display': 'block'},  # Show upload section
                    {'display': 'none'},  # Hide return button
                )

            # Prepare dropdown options
            university_options = [{'label': uni, 'value': uni} for uni in sorted(df['university'].unique())]
            department_options = [{'label': dept, 'value': dept} for dept in sorted(df['department'].unique())]
            job_title_options = [{'label': title, 'value': title} for title in sorted(df['job_title'].unique())]

            if run_demo_state == "demo":
                string = "You are now viewing demo data that has been synthetically generated to demonstrate dashboard functionality. Due to the randomness of the data, the ML functionality might not create accurate predictions. "
            else:
                string = "You are now using your own data"
            # Show dashboard
            return (
                f"{string}",
                {**dashboard_content_style, 'display': 'block'},  # Preserve other styles, change display
                {**page_content_style, 'height': '20vh', 'display': 'block'},  # Preserve other styles, change display
                university_options,
                department_options,
                job_title_options,
                None,
                None,
                run_demo_state,
                {'display': 'none'},  # Hide upload section
                {'display': 'block'},  # Show return button
            )


    # Callback for insights and visualizations (similar to previous implementation)
    @app.callback(
        [Output('insights-container', 'children'),
        Output('performance-by-university', 'figure'),
        Output('salary-distribution', 'figure'),
        Output('experience-salary-scatter', 'figure'), # Added this and next two lines
        Output('salary-department-boxplot', 'figure'),
        Output('performance-heatmap', 'figure'),
        Output('data-table', 'columns'),
        Output('data-table', 'data')],
        [Input('run-demo-state', 'children'),
        Input('upload-data', 'contents'),
        Input('run-demo-btn', 'n_clicks'),
        Input('university-dropdown', 'value'),
        Input('department-dropdown', 'value'),
        Input('job-title-dropdown', 'value')],
        [State('upload-data', 'filename')],
        prevent_initial_call=True
    )
    def update_dashboard(run_demo_state, contents, demo_clicks, selected_university, selected_department, selected_job_title, filename):
        if run_demo_state == 'demo':
            # Load demo data
            demo_csv_path = 'demo_data.csv'
            try:
                df = pd.read_csv(demo_csv_path)
            except Exception as e:
                return f"Error loading demo data: {str(e)}", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], []

        elif contents: # If contents are provided (i.e., a file was uploaded)
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            except Exception as e:
                return f"Error processing file 2: {str(e)}", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], []

        else:
            return "No data available", go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), [], []

        # Apply filters
        filtered_df = df.copy()

        if selected_university:
            filtered_df = filtered_df[filtered_df['university'] == selected_university]

        if selected_department:
            filtered_df = filtered_df[filtered_df['department'] == selected_department]

        if selected_job_title:
            filtered_df = filtered_df[filtered_df['job_title'] == selected_job_title]

        # Handle empty filtered data
        if filtered_df.empty:
            return (
                "No data available for the selected filters.",
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                go.Figure(),
                [],
                []
            )
        # 1. Scatter Plot: Years of Experience vs Salary
        experience_salary_fig = px.scatter(
            filtered_df,
            x='years_of_experience',
            y='salary',
            color='department',
            title='Salary vs Years of Experience',
            labels={'years_of_experience': 'Years of Experience', 'salary': 'Salary ($)'},
            hover_data=['employee_id', 'job_title']
        )
        experience_salary_fig.update_layout(
            xaxis_title='Years of Experience',
            yaxis_title='Salary ($)'
        )

        # 2. Box Plot: Salary Distribution by Department
        salary_department_fig = go.Figure()
        for department in filtered_df['department'].unique():
            dept_data = filtered_df[filtered_df['department'] == department]['salary']
            salary_department_fig.add_trace(
                go.Box(
                    y=dept_data,
                    name=department,
                    boxmean=True  # Add mean marker
                )
            )
        salary_department_fig.update_layout(
            title='Salary Distribution by Department',
            yaxis_title='Salary ($)',
            xaxis_title='Department'
        )

        # 3. Heatmap: Performance Scores Correlation
        # Select numerical columns for correlation
        numerical_columns = ['graduation_grade', 'performance_score',
                            'years_of_experience', 'age', 'salary']
        correlation_matrix = filtered_df[numerical_columns].corr()

        performance_heatmap = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1
        ))
        performance_heatmap.update_layout(
            title='Correlation Heatmap of Performance-Related Attributes',
            xaxis_title='Attributes',
            yaxis_title='Attributes'
        )
        # Insights
        insights = {
            'total_employees': len(filtered_df),
            'average_performance': filtered_df['performance_score'].mean(),
            'average_salary': filtered_df['salary'].mean()
        }

        insights_cards = [
            html.Div(
                className='graph-container',  # Add a class for the container
                children=[
                    html.Div([
                        html.H5('Total Employees'),
                        html.P(f"{insights['total_employees']}")
                    ], className='square-box'),

                    html.Div([
                        html.H5('Avg Performance'),
                        html.P(f"{insights['average_performance']:.2f}")
                    ], className='square-box'),

                    html.Div([
                        html.H5('Avg Salary'),
                        html.P(f"${insights['average_salary']:,.2f}")
                    ], className='square-box')
                ]
            )
        ]

        # Visualizations
        performance_fig = px.bar(
            filtered_df.groupby('university')['performance_score'].mean().reset_index(),
            x='university', y='performance_score', title='Performance by University'
        )

        salary_fig = px.histogram(
            filtered_df, x='salary', title='Salary Distribution'
        )

        # Data Table
        columns = [{"name": i, "id": i} for i in filtered_df.columns]

        return (
            insights_cards,
            performance_fig,
            salary_fig,
            experience_salary_fig,
            salary_department_fig,
            performance_heatmap,
            columns,
            filtered_df.to_dict('records')
        )

    # Callback for ML model insights
    @app.callback(
        [Output('performance-prediction-model-insights', 'children'),
        Output('feature-importance-chart', 'figure'),
        Output('employee-clustering-chart', 'figure')],
        [Input('run-demo-state', 'children'),
        Input('upload-data', 'contents')],
        prevent_initial_call=True
    )
    def update_performance_insights(run_demo_state, contents):
        # Load data
        if run_demo_state == 'demo':
            df = pd.read_csv('demo_data.csv')
        elif contents:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return "No data available", {}, {}

        # Ensure required columns for performance prediction are present
        required_columns = [
            'employee_id', 'university', 'graduation_grade',
            'department', 'job_title', 'start_date',
            'performance_score', 'years_of_experience',
            'age', 'salary'
        ]
        if not all(col in df.columns for col in required_columns):
            return "Missing required columns for performance prediction", {}, {}

        # Generate ML insights for performance prediction
        ml_insights = create_ml_insights(df)

        # Performance Prediction Insights
        performance_insights = [
            html.P(f"Model Training Accuracy: {ml_insights['train_score']:.2%}"),
            html.P(f"Model Testing Accuracy: {ml_insights['test_score']:.2%}")
        ]

        # Feature Importance Visualization
        feature_importance_fig = px.bar(
            ml_insights['feature_importance'],
            x='Feature',
            y='Importance',
            title='Feature Importance for Performance Prediction'
        )

        # Clustering Visualization
        clustering_fig = px.scatter(
            ml_insights['clustering_df'],
            x='PCA1',
            y='PCA2',
            color='Cluster',
            hover_data=['Performance', 'Department'],
            title='Employee Clustering Visualization'
        )

        return performance_insights, feature_importance_fig, clustering_fig

    @app.callback(
        Output('performance-prediction-result', 'children'),
        [Input('predict-performance-btn', 'n_clicks'),
        Input('run-demo-state', 'children'),
        Input('upload-data', 'contents')],
        [State('demo-grade', 'value'),
        State('demo-experience', 'value')],
        prevent_initial_call=True
    )
    def predict_performance(n_clicks, run_demo_state, contents, grade, experience):
        # Load data
        if run_demo_state == 'demo':
            df = pd.read_csv('demo_data.csv')
        elif contents:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return html.P("No data available")

        # Generate ML insights
        ml_insights = create_ml_insights(df)
        
        # Define feature names (same as those used in `create_ml_insights`)
        features = ['graduation_grade', 'years_of_experience', 'age', 'salary']

        # Prepare input for prediction with appropriate feature names
        input_data = pd.DataFrame(
            [[grade, experience, 30, 85000]],  # Example input: age=30, salary=85,000
            columns=features
        )

        # Scale the input data
        input_scaled = ml_insights['scaler'].transform(input_data)
        
        # Predict performance
        predicted_performance = ml_insights['performance_predictor'].predict(input_scaled)[0]
        
        # Return result
        return html.P(f"Predicted Performance Score: {predicted_performance:,.2f}")


    return app


def main():
    # Create and run the Dash app
    app = create_dash_app()
    app.run_server(debug=True)

if __name__ == "__main__":
    main()