%% step15_fault_injection_demo.m
% NEW PROJECT: MATLAB AI Etch Autoencoder
% STEP 15 - Synthetic Fault Injection Sensitivity Demo
%
% 목적
% 1) 기존 모델/threshold를 전혀 수정하지 않고,
%    정상 Test Run의 특정 Sensor에 인위적 shift를 주었을 때
%    3-Sigma / Isolation Forest / Autoencoder Risk가 어떻게 변하는지 확인
% 2) Autoencoder가 injected sensor를 troubleshooting priority로 올리는지 확인
%
% 중요
% - Main Test(STEP09)와 별개의 POST-HOC DEMO
% - 물리 단위(ppm, W, Torr) Fault Injection이 아니라
%   STEP05C 이후의 standardized model-input 공간에서 +k sigma shift를 주는 실험
% - 모델 재학습 / threshold retuning 없음
% - Test 결과를 개선하기 위한 tuning이 아님
%
% 기본 Injection
%   Sensor : BCl3 Flow
%   Step   : 4
%   Window : 4 ~ 6
%   Shift  : 0, 1, 2, 3, 4, 5, 6 sigma
%
% Baseline Run
%   Test Normal 중 Autoencoder와 Isolation Forest가 모두 NORMAL로 판단한
%   첫 Run을 자동 선택
%
% 입력
%   data/processed/10_step4_model_ready.csv
%   data/processed/11_step5_model_ready.csv
%   results/tables/16_model_feature_quality.csv
%   results/tables/23_iforest_thresholds.csv
%   results/tables/27_autoencoder_thresholds.csv
%   results/tables/32_final_test_run_predictions.csv
%   results/tables/42_ae_sensor_normal_reference.csv
%   models/step07_isolation_forest.mat
%   models/step08_autoencoder.mat
%
% 출력
%   results/tables/61_fault_injection_sensitivity.csv
%   results/tables/62_fault_injection_window_detail.csv
%   results/figures/step15_fault_injection_sensitivity.png

clc;
clear;
close all;

fprintf('\n============================================================\n');
fprintf(' NEW MATLAB AI ETCH PROJECT | STEP 15 FAULT INJECTION\n');
fprintf('============================================================\n\n');

%% 0. 설정
INJECT_SENSOR = "BCl3 Flow";
INJECT_STEP = 4;
INJECT_WINDOWS = 4:6;
SHIFT_LEVELS = [0 1 2 3 4 5 6];

SIGMA_THRESHOLD = 3.0;
MINI_BATCH_SIZE = 64;

%% 1. 경로
projectRoot = fileparts(fileparts(mfilename("fullpath")));
processedDir = fullfile(projectRoot,"data","processed");
tableDir = fullfile(projectRoot,"results","tables");
figureDir = fullfile(projectRoot,"results","figures");
modelDir = fullfile(projectRoot,"models");

if ~isfolder(figureDir), mkdir(figureDir); end

step4File = fullfile(processedDir,"10_step4_model_ready.csv");
step5File = fullfile(processedDir,"11_step5_model_ready.csv");

qualityFile = fullfile(tableDir,"16_model_feature_quality.csv");
ifThresholdFile = fullfile(tableDir,"23_iforest_thresholds.csv");
aeThresholdFile = fullfile(tableDir,"27_autoencoder_thresholds.csv");
predictionFile = fullfile(tableDir,"32_final_test_run_predictions.csv");
sensorRefFile = fullfile(tableDir,"42_ae_sensor_normal_reference.csv");

ifModelFile = fullfile(modelDir,"step07_isolation_forest.mat");
aeModelFile = fullfile(modelDir,"step08_autoencoder.mat");

requiredFiles = [ ...
    string(step4File),string(step5File), ...
    string(qualityFile),string(ifThresholdFile), ...
    string(aeThresholdFile),string(predictionFile), ...
    string(sensorRefFile), ...
    string(ifModelFile),string(aeModelFile)];

for i = 1:numel(requiredFiles)
    assert(isfile(requiredFiles(i)), ...
        "필수 파일이 없습니다: %s",requiredFiles(i));
end

%% 2. Load
D4 = readT(step4File);
D5 = readT(step5File);
Q = readT(qualityFile);
IFT = readT(ifThresholdFile);
AET = readT(aeThresholdFile);
P = readT(predictionFile);
SR = readT(sensorRefFile);

IFM = load(ifModelFile,"models");
AEM = load(aeModelFile,"autoencoders");

ifModels = IFM.models;
autoencoders = AEM.autoencoders;

%% 3. Baseline Normal Test Run 자동 선택
candidate = P( ...
    strcmpi(P.Status,"Normal") & ...
    ~logical(P.Pred_AE) & ...
    ~logical(P.Pred_IF),:);

assert(~isempty(candidate), ...
    "AE와 IF가 모두 정상 판정한 Normal Test Run이 없습니다.");

% AE risk가 낮은 순으로 가장 안정적인 baseline 선택
candidate = sortrows(candidate,"Risk_AE","ascend");
BASE_RUN = candidate.Run(1);

fprintf("=== FAULT INJECTION SETUP ===\n");
fprintf("Baseline Run            : %s\n",BASE_RUN);
fprintf("Ground Truth            : Normal\n");
fprintf("Injected Sensor         : %s\n",INJECT_SENSOR);
fprintf("Injected Step           : %d\n",INJECT_STEP);
fprintf("Injected Windows        : %s\n",mat2str(INJECT_WINDOWS));
fprintf("Standardized Shift      : %s sigma\n\n",mat2str(SHIFT_LEVELS));

%% 4. Baseline Run 데이터
R4 = D4(D4.Run==BASE_RUN,:);
R5 = D5(D5.Run==BASE_RUN,:);

assert(height(R4)==9 && height(R5)==9, ...
    "Baseline Run은 Step별 9 windows여야 합니다.");

metaNames = [ ...
    "Run","Experiment","Split","Status","Fault_Name", ...
    "Step","Window_Index","Start_Grid","End_Grid", ...
    "Start_Norm_Time","End_Norm_Time"];

vars4 = string(R4.Properties.VariableNames);
vars5 = string(R5.Properties.VariableNames);

features4 = vars4(~ismember(vars4,metaNames));
features5 = vars5(~ismember(vars5,metaNames));

X4base = double(R4{:,cellstr(features4)});
X5base = double(R5{:,cellstr(features5)});

%% 5. Inject sensor feature 찾기
q4 = Q(Q.Step==4 & Q.Keep_For_Model==1,:);
q5 = Q(Q.Step==5 & Q.Keep_For_Model==1,:);

[tf4,loc4] = ismember(features4,q4.Feature);
[tf5,loc5] = ismember(features5,q5.Feature);

assert(all(tf4) && all(tf5),"Feature-Sensor mapping 실패");

featureSensors4 = q4.Sensor_Name(loc4);
featureSensors5 = q5.Sensor_Name(loc5);

injectFeatureMask4 = featureSensors4==INJECT_SENSOR;
injectFeatureMask5 = featureSensors5==INJECT_SENSOR;

if INJECT_STEP==4
    assert(any(injectFeatureMask4), ...
        "Step4에서 Inject Sensor를 찾지 못했습니다.");
else
    assert(any(injectFeatureMask5), ...
        "Step5에서 Inject Sensor를 찾지 못했습니다.");
end

%% 6. Frozen thresholds / models
ifTh4 = IFT.Frozen_Threshold(IFT.Step==4);
ifTh5 = IFT.Frozen_Threshold(IFT.Step==5);

aeTh4 = AET.Frozen_Threshold(AET.Step==4);
aeTh5 = AET.Frozen_Threshold(AET.Step==5);

assert(numel(ifTh4)==1 && numel(ifTh5)==1);
assert(numel(aeTh4)==1 && numel(aeTh5)==1);

ifM4 = ifModels.Step4.Model;
ifM5 = ifModels.Step5.Model;

aeM4 = autoencoders.Step4.Network;
aeM5 = autoencoders.Step5.Network;

%% 7. Sensor normal-reference
sensorRef4 = SR.Normal_Reference_Max_RMSE( ...
    SR.Step==4 & SR.Sensor_Name==INJECT_SENSOR);

sensorRef5 = SR.Normal_Reference_Max_RMSE( ...
    SR.Step==5 & SR.Sensor_Name==INJECT_SENSOR);

assert(numel(sensorRef4)==1 && numel(sensorRef5)==1, ...
    "Injected Sensor normal reference를 찾지 못했습니다.");

%% 8. Shift sweep
sensitivityRows = table();
windowDetailRows = table();

for level = SHIFT_LEVELS

    X4 = X4base;
    X5 = X5base;

    % standardized model-input 공간에서 +k sigma shift
    if INJECT_STEP==4
        rowMask = ismember(R4.Window_Index,INJECT_WINDOWS);
        X4(rowMask,injectFeatureMask4) = ...
            X4(rowMask,injectFeatureMask4) + level;
    else
        rowMask = ismember(R5.Window_Index,INJECT_WINDOWS);
        X5(rowMask,injectFeatureMask5) = ...
            X5(rowMask,injectFeatureMask5) + level;
    end

    %% 8-1. 3-Sigma
    sigmaScore4 = max(abs(X4),[],2);
    sigmaScore5 = max(abs(X5),[],2);

    sigmaRisk4 = sigmaScore4 / SIGMA_THRESHOLD;
    sigmaRisk5 = sigmaScore5 / SIGMA_THRESHOLD;

    combinedSigmaRisk = max([sigmaRisk4;sigmaRisk5]);
    predSigma = combinedSigmaRisk > 1;

    %% 8-2. Isolation Forest
    [~,ifScore4] = isanomaly(ifM4,X4);
    [~,ifScore5] = isanomaly(ifM5,X5);

    ifRisk4 = ifScore4 / ifTh4;
    ifRisk5 = ifScore5 / ifTh5;

    combinedIFRisk = max([ifRisk4;ifRisk5]);
    predIF = combinedIFRisk > 1;

    %% 8-3. Autoencoder
    X4hat = minibatchpredict( ...
        aeM4,single(X4),MiniBatchSize=MINI_BATCH_SIZE);

    X5hat = minibatchpredict( ...
        aeM5,single(X5),MiniBatchSize=MINI_BATCH_SIZE);

    X4hat = toNumericMatrix(X4hat,X4);
    X5hat = toNumericMatrix(X5hat,X5);

    residual4 = X4-X4hat;
    residual5 = X5-X5hat;

    aeRMSE4 = sqrt(mean(residual4.^2,2));
    aeRMSE5 = sqrt(mean(residual5.^2,2));

    aeRisk4 = aeRMSE4 / aeTh4;
    aeRisk5 = aeRMSE5 / aeTh5;

    combinedAERisk = max([aeRisk4;aeRisk5]);
    predAE = combinedAERisk > 1;

    % AE peak Step / Window
    [step4PeakRisk,idx4] = max(aeRisk4);
    [step5PeakRisk,idx5] = max(aeRisk5);

    if step4PeakRisk >= step5PeakRisk
        peakStep = 4;
        peakWindow = R4.Window_Index(idx4);
        peakRisk = step4PeakRisk;
    else
        peakStep = 5;
        peakWindow = R5.Window_Index(idx5);
        peakRisk = step5PeakRisk;
    end

    %% 8-4. Injected sensor reconstruction risk
    sensorRMSE4 = sqrt(mean(residual4(:,injectFeatureMask4).^2,2));
    sensorRMSE5 = sqrt(mean(residual5(:,injectFeatureMask5).^2,2));

    sensorRisk4 = sensorRMSE4 / sensorRef4;
    sensorRisk5 = sensorRMSE5 / sensorRef5;

    injectedSensorMaxRisk = max([sensorRisk4;sensorRisk5]);

    %% 8-5. 모든 Sensor 중 AE Top Sensor
    [topSensor,topSensorRisk] = getTopSensorRisk( ...
        residual4,residual5, ...
        featureSensors4,featureSensors5, ...
        SR);

    %% 8-6. Summary row
    row = table( ...
        level, ...
        combinedSigmaRisk,predSigma, ...
        combinedIFRisk,predIF, ...
        combinedAERisk,predAE, ...
        peakStep,peakWindow,peakRisk, ...
        injectedSensorMaxRisk, ...
        topSensor,topSensorRisk, ...
        'VariableNames', ...
        {'Injected_Shift_Sigma', ...
         'Risk_3Sigma','Pred_3Sigma', ...
         'Risk_IF','Pred_IF', ...
         'Risk_AE','Pred_AE', ...
         'AE_Peak_Step','AE_Peak_Window','AE_Peak_Risk', ...
         'Injected_Sensor_Max_Risk', ...
         'AE_Top_Sensor','AE_Top_Sensor_Risk'});

    sensitivityRows = [sensitivityRows;row]; %#ok<AGROW>

    %% 8-7. Window detail
    for j = 1:height(R4)
        wd = table( ...
            level,BASE_RUN, ...
            4,R4.Window_Index(j), ...
            R4.Start_Norm_Time(j),R4.End_Norm_Time(j), ...
            aeRisk4(j), ...
            aeRisk4(j)>1, ...
            sensorRisk4(j), ...
            'VariableNames', ...
            {'Injected_Shift_Sigma','Run','Step','Window_Index', ...
             'Start_Norm_Time','End_Norm_Time', ...
             'AE_Risk_Ratio','AE_Anomaly', ...
             'Injected_Sensor_Risk'});

        windowDetailRows = [windowDetailRows;wd]; %#ok<AGROW>
    end

    for j = 1:height(R5)
        wd = table( ...
            level,BASE_RUN, ...
            5,R5.Window_Index(j), ...
            R5.Start_Norm_Time(j),R5.End_Norm_Time(j), ...
            aeRisk5(j), ...
            aeRisk5(j)>1, ...
            sensorRisk5(j), ...
            'VariableNames', ...
            {'Injected_Shift_Sigma','Run','Step','Window_Index', ...
             'Start_Norm_Time','End_Norm_Time', ...
             'AE_Risk_Ratio','AE_Anomaly', ...
             'Injected_Sensor_Risk'});

        windowDetailRows = [windowDetailRows;wd]; %#ok<AGROW>
    end
end

%% 9. 저장
writetable(sensitivityRows, ...
    fullfile(tableDir,"61_fault_injection_sensitivity.csv"));

writetable(windowDetailRows, ...
    fullfile(tableDir,"62_fault_injection_window_detail.csv"));

%% 10. Figure
f = figure("Visible","off");
plot( ...
    sensitivityRows.Injected_Shift_Sigma, ...
    sensitivityRows.Risk_AE, ...
    "-o", ...
    LineWidth=1.5);

hold on;

plot( ...
    sensitivityRows.Injected_Shift_Sigma, ...
    sensitivityRows.Risk_IF, ...
    "-o", ...
    LineWidth=1.5);

plot( ...
    sensitivityRows.Injected_Shift_Sigma, ...
    sensitivityRows.Risk_3Sigma, ...
    "-o", ...
    LineWidth=1.5);

yline(1,"--","Decision Threshold");

xlabel("Injected standardized shift (\sigma)");
ylabel("Run Risk Ratio");
title("Synthetic Fault Injection Sensitivity");
legend("Autoencoder","Isolation Forest","3-Sigma", ...
    Location="best");
grid on;

exportgraphics(f, ...
    fullfile(figureDir,"step15_fault_injection_sensitivity.png"), ...
    Resolution=180);

close(f);

%% 11. 출력
fprintf("=== FAULT INJECTION SENSITIVITY ===\n");
disp(sensitivityRows);

% AE 최초 threshold crossing
crossIdx = find(sensitivityRows.Pred_AE,1,"first");

fprintf("\n============================================================\n");
fprintf(" STEP 15 CHECK\n");
fprintf("============================================================\n");

fprintf("Baseline Run                       : %s\n",BASE_RUN);
fprintf("Injected Sensor                    : %s\n",INJECT_SENSOR);
fprintf("Injected Step / Windows            : %d / %s\n", ...
    INJECT_STEP,mat2str(INJECT_WINDOWS));

fprintf("Baseline AE prediction NORMAL      : %s\n", ...
    string(~sensitivityRows.Pred_AE(1)));

fprintf("Baseline IF prediction NORMAL      : %s\n", ...
    string(~sensitivityRows.Pred_IF(1)));

if isempty(crossIdx)
    fprintf("AE threshold crossing              : not reached\n");
else
    fprintf("AE first threshold crossing        : %.1f sigma\n", ...
        sensitivityRows.Injected_Shift_Sigma(crossIdx));
end

fprintf("Injected sensor becomes AE Top-1?  : %s\n", ...
    string(any(sensitivityRows.AE_Top_Sensor==INJECT_SENSOR)));

fprintf("No model retraining                : true\n");
fprintf("No threshold retuning              : true\n");
fprintf("Main STEP09 result modified        : false\n");
fprintf("Injection space                    : standardized model input\n");

generated = [ ...
    isfile(fullfile(tableDir,"61_fault_injection_sensitivity.csv")), ...
    isfile(fullfile(tableDir,"62_fault_injection_window_detail.csv")), ...
    isfile(fullfile(figureDir,"step15_fault_injection_sensitivity.png"))];

fprintf("All output files generated         : %s\n",string(all(generated)));

fprintf("\n============================================================\n");
fprintf(" STEP 15 COMPLETE\n");
fprintf("============================================================\n");

fprintf("\n해석 원칙:\n");
fprintf("- +k sigma는 standardized input에서의 synthetic sensor shift\n");
fprintf("- 물리 단위의 실제 장비 고장을 재현했다고 주장하지 않음\n");
fprintf("- 목적은 frozen AI monitor의 sensitivity와 troubleshooting response 확인\n");
fprintf("- 이 결과로 Main Test 모델/threshold를 재조정하지 않음\n\n");

fprintf("ChatGPT에 아래를 보내주세요.\n");
fprintf("1) FAULT INJECTION SENSITIVITY\n");
fprintf("2) STEP 15 CHECK\n\n");

%% ================= LOCAL FUNCTIONS =================

function T = readT(file)

opts = detectImportOptions(file,"VariableNamingRule","preserve");
T = readtable(file,opts);

vars = string(T.Properties.VariableNames);

if ismember("Run",vars), T.Run = string(T.Run); end
if ismember("Split",vars), T.Split = string(T.Split); end
if ismember("Status",vars), T.Status = string(T.Status); end
if ismember("Fault_Name",vars), T.Fault_Name = string(T.Fault_Name); end
if ismember("Sensor_Name",vars), T.Sensor_Name = string(T.Sensor_Name); end
if ismember("Feature",vars), T.Feature = string(T.Feature); end
if ismember("Model",vars), T.Model = string(T.Model); end
end

function Xhat = toNumericMatrix(prediction,Xreference)

if isa(prediction,"dlarray")
    prediction = extractdata(prediction);
end

Xhat = double(prediction);

if size(Xhat,1) ~= size(Xreference,1) && ...
        size(Xhat,2)==size(Xreference,1) && ...
        size(Xhat,1)==size(Xreference,2)
    Xhat = Xhat';
end

assert(isequal(size(Xhat),size(Xreference)), ...
    "Autoencoder reconstruction size mismatch");
end

function [topSensor,topRisk] = getTopSensorRisk( ...
    residual4,residual5, ...
    sensorMap4,sensorMap5, ...
    sensorReference)

sensors = unique([sensorMap4(:);sensorMap5(:)],"stable");

topSensor = "";
topRisk = -inf;

for s = 1:numel(sensors)

    sensorName = sensors(s);

    idx4 = sensorMap4==sensorName;
    idx5 = sensorMap5==sensorName;

    if any(idx4)
        rmse4 = sqrt(mean(residual4(:,idx4).^2,2));
        ref4 = sensorReference.Normal_Reference_Max_RMSE( ...
            sensorReference.Step==4 & ...
            sensorReference.Sensor_Name==sensorName);

        if isempty(ref4)
            risk4 = -inf;
        else
            risk4 = max(rmse4/ref4);
        end
    else
        risk4 = -inf;
    end

    if any(idx5)
        rmse5 = sqrt(mean(residual5(:,idx5).^2,2));
        ref5 = sensorReference.Normal_Reference_Max_RMSE( ...
            sensorReference.Step==5 & ...
            sensorReference.Sensor_Name==sensorName);

        if isempty(ref5)
            risk5 = -inf;
        else
            risk5 = max(rmse5/ref5);
        end
    else
        risk5 = -inf;
    end

    thisRisk = max(risk4,risk5);

    if thisRisk > topRisk
        topRisk = thisRisk;
        topSensor = sensorName;
    end
end
end
