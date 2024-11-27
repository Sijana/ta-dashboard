import dash
from dash import dcc, html, dash_table, callback
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import io
import base64

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
        'graduation_gpa': [3.8, 3.9, 3.7],
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
            # Similar to previous implementation - filters, insights, etc.
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

                # Salary range filter
                # Salary range filter (without min and max set)
                html.Div([
                    html.Label('Filter By Salary Range'),
                    dcc.RangeSlider(
                        id='salary-range-slider',
                        step=1000,
                        marks={i: f"${i:,}" for i in range(0, 100001, 10000)},  # marks for the range slider
                        value=[20000, 80000]  # default value for the salary range
                    )
                ], className='filter-section'),

            ], className='filter-container'),
            
            # Hidden Demo State Container
            html.Div(id='run-demo-state', style={'display': 'none'}),
            
            # Insights Container
            html.Div(id='insights-container', className='insights-container'),
            
            # Visualizations
            html.Div([
                html.Div([
                    dcc.Graph(id='performance-by-university')
                ], className='graph-container'),
                
                html.Div([
                    dcc.Graph(id='salary-distribution')
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
            'employee_id', 'university', 'graduation_gpa',
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
        Output('data-table', 'columns'),
        Output('data-table', 'data'),
        Output('salary-range-slider', 'min'),
        Output('salary-range-slider', 'max')],
        [Input('run-demo-state', 'children'),
        Input('upload-data', 'contents'),
        Input('run-demo-btn', 'n_clicks'),
        Input('university-dropdown', 'value'),
        Input('department-dropdown', 'value'),
        Input('job-title-dropdown', 'value'),
        Input('salary-range-slider', 'value')],
        [State('upload-data', 'filename')],
        prevent_initial_call=True
    )
    def update_dashboard(run_demo_state, contents, demo_clicks, selected_university, selected_department, selected_job_title, salary_range, filename):
        if run_demo_state == 'demo':
            demo_csv_path = 'demo_data.csv'
            try:
                df = pd.read_csv(demo_csv_path)
            except Exception as e:
                return f"Error loading demo data: {str(e)}", {}, {}, [], [], 0, 0

        elif contents:
            try:
                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            except Exception as e:
                return f"Error processing file 2: {str(e)}", {}, {}, [], [], 0, 0

        else:
            return "No data available 1", {}, {}, [], [], 0, 0

        # Determine which filter was changed
        filter_changed = any([selected_university, selected_department, selected_job_title])

        # Apply filters if any filter has changed
        filtered_df = df.copy()

        if selected_university:
            filtered_df = filtered_df[filtered_df['university'] == selected_university]

        if selected_department:
            filtered_df = filtered_df[filtered_df['department'] == selected_department]

        if selected_job_title:
            filtered_df = filtered_df[filtered_df['job_title'] == selected_job_title]

        if salary_range:
            min_salary, max_salary = salary_range
            filtered_df = filtered_df[(filtered_df['salary'] >= min_salary) & (filtered_df['salary'] <= max_salary)]

        # Handle empty filtered data
        if filtered_df.empty:
            return (
                "No data available for the selected filters.",
                go.Figure(),
                go.Figure(),
                [],
                [],
                0, 0  # Return default min and max for salary
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

        # Return the updated values, including the dynamically set salary min and max
        return (
            insights_cards,
            performance_fig,
            salary_fig,
            columns,
            filtered_df.to_dict('records'),
            filtered_df['salary'].min(),  # dynamically set min salary
            filtered_df['salary'].max()   # dynamically set max salary
        )

    
    return app

def main():
    # Create and run the Dash app
    app = create_dash_app()
    app.run_server(debug=True)

if __name__ == "__main__":
    main()