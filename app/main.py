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


@app.get("/")
def root():
      return {"message": "Welcome to Artificial Intelligence"}
    

@app.post("/predict_employee")
def predict(employee:Employee):
    age=employee.Age
    bussinessTravel=employee.BusinessTravel
    DailyRate=employee.DailyRate
    DistanceFromHome=employee.DistanceFromHome
    Education=employee.Education
    EnvironmentSatisfaction=employee.EnvironmentSatisfaction
    Gender=employee.Gender
    HourlyRate=employee.HourlyRate
    JobInvolvement=employee.JobInvolvement
    JobLevel=employee.JobLevel
    JobSatisfaction=employee.JobSatisfaction
    MonthlyIncome=employee.MonthlyIncome
    MonthlyRate=employee.MonthlyRate
    NumCompaniesWorked=employee.NumCompaniesWorked
    OverTime=employee.OverTime
    PercentSalaryHike=employee.PercentSalaryHike
    PerformanceRating=employee.PerformanceRating
    RelationshipSatisfaction=employee.RelationshipSatisfaction
    StockOptionLevel=employee.StockOptionLevel
    TotalWorkingYears=employee.TotalWorkingYears
    TrainingTimesLastYear=employee.TrainingTimesLastYear
    WorkLifeBalance=employee.WorkLifeBalance
    YearsAtCompany=employee.YearsAtCompany
    YearsInCurrentRole=employee.YearsInCurrentRole
    YearsSinceLastPromotion=employee.YearsSinceLastPromotion
    YearsWithCurrManager=employee.YearsWithCurrManager
    Department_HumanResources=employee.DepartmentHumanResources
    Department_ResearchDevelopment=employee.DepartmentResearchDevelopment
    Department_Sales=employee.DepartmentSales
    EducationField_HumanResources=employee.EducationFieldHumanResources
    EducationField_LifeSciences=employee.EducationFieldLifeSciences
    EducationField_Marketing=employee.EducationFieldMarketing
    EducationField_Medical=employee.EducationFieldMedical
    EducationField_Other=employee.EducationFieldOther
    EducationField_TechnicalDegree=employee.EducationFieldTechnicalDegree
    JobRole_HealthcareRepresentative=employee.JobRoleHealthcareRepresentative
    JobRole_HumanResources=employee.JobRoleHumanResources
    JobRole_LaboratoryTechnician=employee.JobRoleLaboratoryTechnician
    JobRole_Manager=employee.JobRoleManager
    JobRole_ManufacturingDirector=employee.JobRoleManufacturingDirector
    JobRole_ResearchDirector=employee.JobRoleResearchDirector
    JobRole_ResearchScientist=employee.JobRoleResearchScientist
    JobRole_SalesExecutive=employee.JobRoleSalesExecutive
    JobRole_SalesRepresentative=employee.JobRoleSalesRepresentative
    MaritalStatus_Divorced=employee.MaritalStatusDivorced
    MaritalStatus_Married=employee.MaritalStatusMarried
    MaritalStatus_Single=employee.MaritalStatusSingle

    features=[[age,bussinessTravel,DailyRate,
DistanceFromHome,
Education,
EnvironmentSatisfaction,
Gender,
HourlyRate,
JobInvolvement,
JobLevel,
JobSatisfaction,
MonthlyIncome,
MonthlyRate,
NumCompaniesWorked,
OverTime,
PercentSalaryHike,
PerformanceRating,
RelationshipSatisfaction,
StockOptionLevel,
TotalWorkingYears,
TrainingTimesLastYear,
WorkLifeBalance,
YearsAtCompany,
YearsInCurrentRole,
YearsSinceLastPromotion,
YearsWithCurrManager,
Department_HumanResources,
Department_ResearchDevelopment,
Department_Sales,
EducationField_HumanResources,
EducationField_LifeSciences,
EducationField_Marketing,
EducationField_Medical,
EducationField_Other,
EducationField_TechnicalDegree,
JobRole_HealthcareRepresentative,
JobRole_HumanResources,
JobRole_LaboratoryTechnician,
JobRole_Manager,
JobRole_ManufacturingDirector,
JobRole_ResearchDirector,
JobRole_ResearchScientist,
JobRole_SalesExecutive,
JobRole_SalesRepresentative,
MaritalStatus_Divorced,
MaritalStatus_Married,
MaritalStatus_Single]]
    
    
    
    modelo=pickle.load(open('./app/model.pkl', 'rb'))
    pred=modelo.predict(features)
    return pred


