from fastapi import FastAPI,Response,status,HTTPException,Query
from fastapi.params import Body
from typing import List
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np

# aplication instance
app = FastAPI()
# request body pydantic model
class Employee(BaseModel):
    Age :float
    BusinessTravel:float
    DailyRate:float
    DistanceFromHome:float
    Education:float
    EnvironmentSatisfaction:float
    Gender:float
    HourlyRate:float
    JobInvolvement:float
    JobLevel:float
    JobSatisfaction:float
    MonthlyIncome:float
    MonthlyRate:float
    NumCompaniesWorked:float
    OverTime:float
    PercentSalaryHike:float
    PerformanceRating:float
    RelationshipSatisfaction:float
    StockOptionLevel:float
    TotalWorkingYears:float
    TrainingTimesLastYear:float
    WorkLifeBalance:float
    YearsAtCompany:float
    YearsInCurrentRole:float
    YearsSinceLastPromotion:float
    YearsWithCurrManager:float
    DepartmentHumanResources:float
    DepartmentResearchDevelopment:float
    DepartmentSales:float
    EducationFieldHumanResources:float
    EducationFieldLifeSciences:float
    EducationFieldMarketing:float
    EducationFieldMedical:float
    EducationFieldOther:float
    EducationFieldTechnicalDegree:float
    JobRoleHealthcareRepresentative:float
    JobRoleHumanResources:float
    JobRoleLaboratoryTechnician:float
    JobRoleManager:float
    JobRoleManufacturingDirector:float
    JobRoleResearchDirector:float
    JobRoleResearchScientist:float
    JobRoleSalesExecutive:float
    JobRoleSalesRepresentative:float
    MaritalStatusDivorced:float
    MaritalStatusMarried:float
    MaritalStatusSingle:float

with open('./app/model.pkl','rb') as f:
    model=pickle.load(f)

@app.get("/")
def root():
      return {"message": "Welcome to Artificial Intelligence"}
    

@app.post("/predict_employee")
def predict(employee:Employee):
    df=pd.DataFrame([employee.dict().values()],columns=employee.dict().keys())

    # modelo=pickle.load(open('./app/model.pkl', 'rb'))
    pred=model.predict(df)
    return {"prediction" :int(pred)}