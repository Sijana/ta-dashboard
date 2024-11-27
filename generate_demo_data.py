import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_hiring_data(num_employees=100):
    """
    Generate a synthetic dataset of hiring information
    
    Args:
        num_employees (int): Number of employees to generate data for
    
    Returns:
        pandas.DataFrame: Synthetic hiring dataset
    """
    # Seed for reproducibility
    np.random.seed(42)
    
    # List of universities
    universities = [
        'Stanford University', 'MIT', 'Harvard University', 
        'UC Berkeley', 'Carnegie Mellon', 'Princeton University', 
        'CalTech', 'Georgia Tech', 'Cornell University', 
        'University of Michigan'
    ]
    
    # Departments
    departments = [
        'Engineering', 'Sales', 'Marketing', 'Product', 
        'Customer Support', 'Finance', 'HR', 'Data Science'
    ]
    
    # Job Titles
    job_titles = [
        'Junior Engineer', 'Senior Engineer', 'Data Analyst', 
        'Sales Representative', 'Marketing Specialist', 
        'Product Manager', 'Customer Support Specialist', 
        'Financial Analyst', 'HR Coordinator', 'Data Scientist'
    ]
    
    # Generate data
    data = {
        'employee_id': range(1, num_employees + 1),
        'university': np.random.choice(universities, num_employees),
        'graduation_gpa': np.round(np.random.uniform(2.5, 4.0, num_employees), 2),
        'department': np.random.choice(departments, num_employees),
        'job_title': np.random.choice(job_titles, num_employees),
        'start_date': [datetime.now() - timedelta(days=np.random.randint(0, 365*5)) for _ in range(num_employees)],
        'performance_score': np.round(np.random.normal(75, 10, num_employees), 2),
        'years_of_experience': np.round(np.random.uniform(0, 10, num_employees), 1),
        'age': np.random.randint(22, 50, num_employees),
        'salary': np.round(np.random.normal(75000, 25000, num_employees), 2)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure performance scores are between 0 and 100
    df['performance_score'] = df['performance_score'].clip(0, 100)
    
    # Format start date
    df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%d')
    
    return df

def main():
    # Generate dataset
    hiring_data = generate_hiring_data(500)  # Generate 500 employee records
    
    # Save to CSV
    output_path = 'demo_data.csv'
    hiring_data.to_csv(output_path, index=False)
    print(f"Synthetic hiring data generated and saved to {output_path}")
    
    # Display sample of the data
    print("\nData Sample:")
    print(hiring_data.head())
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(hiring_data.describe())

if __name__ == "__main__":
    main()