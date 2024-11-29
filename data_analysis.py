import dash
from dash import dcc, html, dash_table, callback
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

def create_dash_app():
    """
    Create Dash application with upload and download functionality
    
    Returns:
        dash.Dash: Configured Dash application
    """
        
    app = dash.Dash(__name__, external_stylesheets=[
        'https://codepen.io/chriddyp/pen/bWLwgP.css',  # Basic Dash CSS
        '/assets/custom.css'  # Custom CSS file
    ])
    
    app.layout = html.Div([
        html.H1('Hiring Data Dashboard', className='main-title'),
        
        # Instructions paragraph
        html.P(
            """
            Welcome to the Hiring Data Dashboard! You can upload your own CSV file or use the demo data
            to explore insights about hiring trends. Use the filters to narrow down the data, and interact
            with the visualizations to analyze performance, salaries, and more.
            """,
            className='instructions'
        ),

        # File Upload and Download Section
        html.Div([
            # Download Sample CSV Button
            html.Div([
                html.Button('Download Template CSV', id='download-sample-btn', n_clicks=0, className='btn-primary'),
                dcc.Download(id='download-sample-csv')
            ], className='upload-section'),
            
            # CSV File Upload
            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ], className='upload-text'),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center'
                    },
                    multiple=False
                )
            ], className='upload-section'),
            html.Div([
                html.Button('Run Demo', id='run-demo-btn', n_clicks=0, className='btn-secondary')
            ], className='upload-section'),   
        ], className='upload-container'),
        
        # Validation Message
        html.Div(id='upload-validation', className='validation-message'),
        
        # Filters and Dashboard (Initially Hidden)
        html.Div(id='dashboard-content', style={'display': 'none'}, children=[
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
            
            # 1st Row Visualizations
            html.Div([
                html.Div([
                    dcc.Graph(id='performance-by-university')
                ], className='graph-container'),

                html.Div([
                    dcc.Graph(id='salary-distribution')
                ], className='graph-container')
            ], className='visualization-container'),
            
            # 2nd Row Visualizations
            html.Div([
                # New visualizations
                html.Div([
                    dcc.Graph(id='experience-salary-scatter')
                ], className='graph-container'),

                html.Div([
                    dcc.Graph(id='salary-department-boxplot')
                ], className='graph-container'),

                html.Div([
                    dcc.Graph(id='performance-heatmap')
                ], className='graph-container')
            ], className='visualization-container'),

            # Data Table
            html.Div([
                html.H3('Filtered Employee Data'),
                dash_table.DataTable(
                    id='data-table',
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'}
                )
            ], className='table-container')
        ])
    ])
    
    # Callback for downloading sample CSV
    @app.callback(
        Output('download-sample-csv', 'data'),
        Input('download-sample-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_sample_csv(n_clicks):
        sample_df = generate_sample_csv()
        return dcc.send_data_frame(sample_df.to_csv, 'sample_hiring_data.csv', index=False)
    
    # Callback for handling file upload
    @app.callback(
        [Output('upload-validation', 'children'),
        Output('dashboard-content', 'style'),
        Output('university-dropdown', 'options'),
        Output('department-dropdown', 'options'),
        Output('job-title-dropdown', 'options'),
        Output('university-dropdown', 'value'),
        Output('department-dropdown', 'value'),
        Output('run-demo-state', 'children')],
        [Input('upload-data', 'contents'),
        Input('run-demo-btn', 'n_clicks')],
        [State('upload-data', 'filename')],
        prevent_initial_call=True
    )
    def process_data(contents, demo_clicks, filename):
        ctx = dash.callback_context

        # Determine which button triggered the callback
        if not ctx.triggered:
            return '', {'display': 'none'}, [], [], None, None

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        # Initialize run-demo-state as None (for the case no button was clicked)
        run_demo_state = None

        # If "Run Demo" was clicked
        if trigger_id == 'run-demo-btn':
            demo_csv_path = 'demo_data.csv'
            
            try:
                df = pd.read_csv(demo_csv_path)
                run_demo_state = 'demo'
            except Exception as e:
                return f"Error loading demo data: {str(e)}", {'display': 'none'}, [], [], None, None

        # If a file was uploaded
        elif trigger_id == 'upload-data':
            if not contents:
                return '', {'display': 'none'}, [], [], None, None
            
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            except Exception as e:
                return f"Error processing file 1: {str(e)}", {'display': 'none'}, [], [], None, None

        # Validate data
        required_columns = [
            'employee_id', 'university', 'graduation_grade',
            'department', 'job_title', 'start_date',
            'performance_score', 'years_of_experience',
            'age', 'salary'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return f"Missing columns: {', '.join(missing_columns)}", {'display': 'none'}, [], [], None, None

        # Prepare dropdown options
        university_options = [{'label': uni, 'value': uni} for uni in sorted(df['university'].unique())]
        department_options = [{'label': dept, 'value': dept} for dept in sorted(df['department'].unique())]
        job_title_options = [{'label': title, 'value': title} for title in sorted(df['job_title'].unique())]

        return (
            f"Successfully processed data from {trigger_id.replace('-', ' ')}.",
            {'display': 'block'},  # Show dashboard
            university_options,
            department_options,
            job_title_options,
            None,  # Default for university
            None, # Default for department
            run_demo_state
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
                return f"Error loading demo data: {str(e)}", {}, {}, [], []

        elif contents:  # If contents are provided (i.e., a file was uploaded)
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            except Exception as e:
                return f"Error processing file 2: {str(e)}", {}, {}, [], []

        else:
            return "No data available 1", {}, {}, [], []

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
            html.Div([
                html.H4('Total Employees'),
                html.P(f"{insights['total_employees']}")
            ]),
            html.Div([
                html.H4('Avg Performance'),
                html.P(f"{insights['average_performance']:.2f}")
            ]),
            html.Div([
                html.H4('Avg Salary'),
                html.P(f"${insights['average_salary']:,.2f}")
            ])
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
    
    return app

def main():
    # Create and run the Dash app
    app = create_dash_app()
    app.run_server(debug=True)

if __name__ == "__main__":
    main()