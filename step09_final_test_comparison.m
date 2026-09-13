%% step09_final_test_comparison.m
% NEW PROJECT: MATLAB AI Etch Autoencoder
% STEP 09 - Final Sealed Test Comparison
%
% 목적
% 1) 지금까지 봉인했던 Test 42 Runs의 정답을 처음으로 결합
% 2) 3-Sigma vs Isolation Forest vs Autoencoder를 동일 Test Set에서 비교
% 3) Run-level 성능을 중심으로 평가
% 4) Window-level 성능은 보조 지표로만 사용
% 5) AUROC는 각 모델의 continuous risk score로 계산
% 6) False Alarm / Missed Fault Run을 모델별로 확인
%
% Test Set
%   Normal 22 Runs
%   Fault  20 Runs
%
% 입력
%   data/processed/06_test_runs.csv
%   results/tables/20_3sigma_run_scores.csv
%   results/tables/25_iforest_run_scores.csv
%   results/tables/29_autoencoder_run_scores.csv
%   results/tables/19_3sigma_window_scores.csv
%   results/tables/24_iforest_window_scores.csv
%   results/tables/28_autoencoder_window_scores.csv
%
% 출력
%   results/tables/31_final_test_model_comparison.csv
%   results/tables/32_final_test_run_predictions.csv
%   results/tables/33_final_test_fault_detection_detail.csv
%   results/tables/34_final_test_false_alarms.csv
%   results/tables/35_final_test_missed_faults.csv
%   results/tables/36_final_test_window_level_comparison.csv
%   results/tables/37_final_test_roc_points.csv
%
% 중요
% - 이번 STEP에서 처음으로 Test label을 성능 계산에 사용함
% - 이후 모델/threshold를 Test 결과에 맞춰 다시 수정하면 안 됨
% - 개선 실험을 하려면 별도 "post-hoc experiment"로 명확히 분리해야 함

clc;
clear;
close all;

fprintf('\n============================================================\n');
fprintf(' NEW MATLAB AI ETCH PROJECT | STEP 09 FINAL TEST COMPARISON\n');
fprintf('============================================================\n\n');

fprintf("TEST SET IS NOW UNSEALED.\n");
fprintf("From this point, Test labels may be used for final evaluation only.\n");
fprintf("Do NOT retune frozen models/thresholds using these results.\n\n");

%% 0. 경로
projectRoot  = fileparts(fileparts(mfilename("fullpath")));
processedDir = fullfile(projectRoot,"data","processed");
tableDir     = fullfile(projectRoot,"results","tables");

testMetaFile = fullfile(processedDir,"06_test_runs.csv");

sigmaRunFile = fullfile(tableDir,"20_3sigma_run_scores.csv");
ifRunFile    = fullfile(tableDir,"25_iforest_run_scores.csv");
aeRunFile    = fullfile(tableDir,"29_autoencoder_run_scores.csv");

sigmaWinFile = fullfile(tableDir,"19_3sigma_window_scores.csv");
ifWinFile    = fullfile(tableDir,"24_iforest_window_scores.csv");
aeWinFile    = fullfile(tableDir,"28_autoencoder_window_scores.csv");

requiredFiles = [ ...
    string(testMetaFile), ...
    string(sigmaRunFile), ...
    string(ifRunFile), ...
    string(aeRunFile), ...
    string(sigmaWinFile), ...
    string(ifWinFile), ...
    string(aeWinFile)];

for i = 1:numel(requiredFiles)
    assert(isfile(requiredFiles(i)), ...
        "필수 파일이 없습니다: %s",requiredFiles(i));
end

%% 1. Test metadata load
opts = detectImportOptions(testMetaFile,"VariableNamingRule","preserve");
Tmeta = readtable(testMetaFile,opts);

Tmeta.Run = string(Tmeta.Run);
Tmeta.Status = string(Tmeta.Status);
Tmeta.Fault_Name = string(Tmeta.Fault_Name);

testMask = strcmpi(Tmeta.Status,"Normal") | strcmpi(Tmeta.Status,"Fault");
Tmeta = Tmeta(testMask,:);

assert(height(Tmeta)==42, ...
    "Test Run 수가 예상(42)과 다릅니다.");

assert(sum(strcmpi(Tmeta.Status,"Normal"))==22, ...
    "Test Normal Run 수가 예상(22)과 다릅니다.");

assert(sum(strcmpi(Tmeta.Status,"Fault"))==20, ...
    "Test Fault Run 수가 예상(20)과 다릅니다.");

isFaultTruth = strcmpi(Tmeta.Status,"Fault");

fprintf("=== TEST SET ===\n");
fprintf("Total  : %d Runs\n",height(Tmeta));
fprintf("Normal : %d Runs\n",sum(~isFaultTruth));
fprintf("Fault  : %d Runs\n\n",sum(isFaultTruth));

%% 2. Run score load
S = readScoreTable(sigmaRunFile);
I = readScoreTable(ifRunFile);
A = readScoreTable(aeRunFile);

S = S(S.Split=="Test",:);
I = I(I.Split=="Test",:);
A = A(A.Split=="Test",:);

assert(height(S)==42 && height(I)==42 && height(A)==42, ...
    "세 모델의 Test Run score 수가 모두 42여야 합니다.");

%% 3. Test metadata 순서로 score 정렬
[tfS,locS] = ismember(Tmeta.Run,S.Run);
[tfI,locI] = ismember(Tmeta.Run,I.Run);
[tfA,locA] = ismember(Tmeta.Run,A.Run);

assert(all(tfS) && all(tfI) && all(tfA), ...
    "일부 Test Run score를 찾지 못했습니다.");

S = S(locS,:);
I = I(locI,:);
A = A(locA,:);

assert(all(S.Run==Tmeta.Run) && all(I.Run==Tmeta.Run) && all(A.Run==Tmeta.Run), ...
    "Run 정렬에 문제가 있습니다.");

%% 4. 모델별 binary prediction + continuous score
predSigma = logical(S.Run_Anomaly_3Sigma);
predIF    = logical(I.Run_Anomaly_IF);
predAE    = logical(A.Run_Anomaly_AE);

% Continuous risk score
% 3sigma는 threshold 3.0 대비 최대 |Z| 비율
riskSigma = double(S.Combined_MaxScore) / 3.0;

% IF / AE는 STEP07/08에서 이미 threshold ratio로 저장
riskIF = double(I.Combined_RiskRatio);
riskAE = double(A.Combined_RiskRatio);

assert(all(isfinite(riskSigma)) && ...
       all(isfinite(riskIF)) && ...
       all(isfinite(riskAE)), ...
       "Run-level risk score에 NaN/Inf가 있습니다.");

%% 5. Run-level metrics
models = ["3-Sigma";"Isolation Forest";"Autoencoder"];
preds = {predSigma,predIF,predAE};
risks = {riskSigma,riskIF,riskAE};

comparisonRows = table();
rocTables = cell(numel(models),1);

for m = 1:numel(models)

    pred = preds{m};
    score = risks{m};

    TP = sum(pred & isFaultTruth);
    FN = sum(~pred & isFaultTruth);
    FP = sum(pred & ~isFaultTruth);
    TN = sum(~pred & ~isFaultTruth);

    Recall = safeDiv(TP,TP+FN);
    Precision = safeDiv(TP,TP+FP);
    Specificity = safeDiv(TN,TN+FP);
    FPR = safeDiv(FP,FP+TN);
    F1 = safeDiv(2*Precision*Recall,Precision+Recall);
    Accuracy = safeDiv(TP+TN,TP+TN+FP+FN);

    % AUROC
    [fprPts,tprPts,~,auc] = perfcurve(isFaultTruth,score,true);

    rocTables{m} = table( ...
        repmat(models(m),numel(fprPts),1), ...
        fprPts(:), ...
        tprPts(:), ...
        'VariableNames',{'Model','FPR','TPR'});

    row = table( ...
        models(m), ...
        TP,FN,FP,TN, ...
        Recall,Precision,F1,Specificity,FPR,Accuracy,auc, ...
        'VariableNames', ...
        {'Model','TP','FN','FP','TN', ...
         'Recall','Precision','F1','Specificity','FPR','Accuracy','AUROC'});

    comparisonRows = [comparisonRows;row]; %#ok<AGROW>
end

%% 6. Run prediction detail
runPredictions = table( ...
    Tmeta.Run, ...
    Tmeta.Experiment, ...
    Tmeta.Status, ...
    Tmeta.Fault_Name, ...
    isFaultTruth, ...
    predSigma, ...
    riskSigma, ...
    predIF, ...
    riskIF, ...
    predAE, ...
    riskAE, ...
    'VariableNames', ...
    {'Run','Experiment','Status','Fault_Name','Is_Fault_Truth', ...
     'Pred_3Sigma','Risk_3Sigma', ...
     'Pred_IF','Risk_IF', ...
     'Pred_AE','Risk_AE'});

%% 7. Fault detail
faultDetail = runPredictions(isFaultTruth,:);

faultDetail = sortrows(faultDetail, ...
    ["Experiment","Run"]);

%% 8. False alarms
falseAlarmRows = table();

for m = 1:numel(models)

    pred = preds{m};
    score = risks{m};

    idx = ~isFaultTruth & pred;

    if any(idx)
        tmp = table( ...
            repmat(models(m),sum(idx),1), ...
            Tmeta.Run(idx), ...
            Tmeta.Experiment(idx), ...
            score(idx), ...
            'VariableNames', ...
            {'Model','Run','Experiment','Risk_Score'});

        falseAlarmRows = [falseAlarmRows;tmp]; %#ok<AGROW>
    end
end

%% 9. Missed faults
missedRows = table();

for m = 1:numel(models)

    pred = preds{m};
    score = risks{m};

    idx = isFaultTruth & ~pred;

    if any(idx)
        tmp = table( ...
            repmat(models(m),sum(idx),1), ...
            Tmeta.Run(idx), ...
            Tmeta.Experiment(idx), ...
            Tmeta.Fault_Name(idx), ...
            score(idx), ...
            'VariableNames', ...
            {'Model','Run','Experiment','Fault_Name','Risk_Score'});

        missedRows = [missedRows;tmp]; %#ok<AGROW>
    end
end

%% 10. Window-level 보조 비교
% 각 Window의 true label은 해당 Run의 Status를 그대로 부여.
% Window overlap이 있으므로 최종 핵심 성능은 Run-level.
WS = readScoreTable(sigmaWinFile);
WI = readScoreTable(ifWinFile);
WA = readScoreTable(aeWinFile);

WS = WS(WS.Split=="Test",:);
WI = WI(WI.Split=="Test",:);
WA = WA(WA.Split=="Test",:);

windowComparison = table();

windowModels = ["3-Sigma";"Isolation Forest";"Autoencoder"];

for m = 1:3

    if m==1
        W = WS;
        predName = "Anomaly_3Sigma";
    elseif m==2
        W = WI;
        predName = "Anomaly_IF";
    else
        W = WA;
        predName = "Anomaly_AE";
    end

    [tf,loc] = ismember(W.Run,Tmeta.Run);
    assert(all(tf),"Window score에 Test metadata가 없는 Run이 있습니다.");

    truthW = isFaultTruth(loc);
    predW = logical(W.(char(predName)));

    TP = sum(predW & truthW);
    FN = sum(~predW & truthW);
    FP = sum(predW & ~truthW);
    TN = sum(~predW & ~truthW);

    Recall = safeDiv(TP,TP+FN);
    Precision = safeDiv(TP,TP+FP);
    F1 = safeDiv(2*Precision*Recall,Precision+Recall);
    FPR = safeDiv(FP,FP+TN);

    row = table( ...
        windowModels(m), ...
        numel(predW), ...
        TP,FN,FP,TN, ...
        Recall,Precision,F1,FPR, ...
        'VariableNames', ...
        {'Model','Window_Count','TP','FN','FP','TN', ...
         'Recall','Precision','F1','FPR'});

    windowComparison = [windowComparison;row]; %#ok<AGROW>
end

%% 11. 저장
writetable(comparisonRows, ...
    fullfile(tableDir,"31_final_test_model_comparison.csv"));

writetable(runPredictions, ...
    fullfile(tableDir,"32_final_test_run_predictions.csv"));

writetable(faultDetail, ...
    fullfile(tableDir,"33_final_test_fault_detection_detail.csv"));

writetable(falseAlarmRows, ...
    fullfile(tableDir,"34_final_test_false_alarms.csv"));

writetable(missedRows, ...
    fullfile(tableDir,"35_final_test_missed_faults.csv"));

writetable(windowComparison, ...
    fullfile(tableDir,"36_final_test_window_level_comparison.csv"));

rocPoints = vertcat(rocTables{:});

writetable(rocPoints, ...
    fullfile(tableDir,"37_final_test_roc_points.csv"));

%% 12. 출력
fprintf("============================================================\n");
fprintf(" FINAL RUN-LEVEL TEST COMPARISON\n");
fprintf("============================================================\n");
disp(comparisonRows);

fprintf("=== FAULT DETECTION DETAIL ===\n");
disp(faultDetail(:, ...
    ["Run","Experiment","Fault_Name", ...
     "Pred_3Sigma","Risk_3Sigma", ...
     "Pred_IF","Risk_IF", ...
     "Pred_AE","Risk_AE"]));

fprintf("=== FALSE ALARMS ===\n");
if isempty(falseAlarmRows)
    fprintf("False alarm 없음\n");
else
    disp(falseAlarmRows);
end

fprintf("=== MISSED FAULTS ===\n");
if isempty(missedRows)
    fprintf("Missed fault 없음\n");
else
    disp(missedRows);
end

fprintf("=== WINDOW-LEVEL SUPPORTING COMPARISON ===\n");
disp(windowComparison);

%% 13. Best-by-metric 출력
[~,bestF1Idx] = max(comparisonRows.F1);
[~,bestRecallIdx] = max(comparisonRows.Recall);
[~,bestFPRIdx] = min(comparisonRows.FPR);
[~,bestAUCIdx] = max(comparisonRows.AUROC);

fprintf("\n============================================================\n");
fprintf(" STEP 09 CHECK\n");
fprintf("============================================================\n");

fprintf("Test Runs evaluated                 : %d\n",height(Tmeta));
fprintf("Test Normal / Fault                 : %d / %d\n", ...
    sum(~isFaultTruth),sum(isFaultTruth));

fprintf("Best F1 model                       : %s (%.4f)\n", ...
    comparisonRows.Model(bestF1Idx), ...
    comparisonRows.F1(bestF1Idx));

fprintf("Best Recall model                   : %s (%.4f)\n", ...
    comparisonRows.Model(bestRecallIdx), ...
    comparisonRows.Recall(bestRecallIdx));

fprintf("Lowest FPR model                    : %s (%.4f)\n", ...
    comparisonRows.Model(bestFPRIdx), ...
    comparisonRows.FPR(bestFPRIdx));

fprintf("Best AUROC model                    : %s (%.4f)\n", ...
    comparisonRows.Model(bestAUCIdx), ...
    comparisonRows.AUROC(bestAUCIdx));

fprintf("Test labels now officially unsealed : true\n");

generated = [ ...
    isfile(fullfile(tableDir,"31_final_test_model_comparison.csv")), ...
    isfile(fullfile(tableDir,"32_final_test_run_predictions.csv")), ...
    isfile(fullfile(tableDir,"33_final_test_fault_detection_detail.csv")), ...
    isfile(fullfile(tableDir,"34_final_test_false_alarms.csv")), ...
    isfile(fullfile(tableDir,"35_final_test_missed_faults.csv")), ...
    isfile(fullfile(tableDir,"36_final_test_window_level_comparison.csv")), ...
    isfile(fullfile(tableDir,"37_final_test_roc_points.csv"))];

fprintf("All output files generated          : %s\n",string(all(generated)));

fprintf("\n============================================================\n");
fprintf(" STEP 09 COMPLETE\n");
fprintf("============================================================\n");

fprintf("\n중요:\n");
fprintf("- 이 결과가 Main Test Result임\n");
fprintf("- 지금부터 이 Test 결과를 보고 기존 모델/threshold를 재조정하면 안 됨\n");
fprintf("- 추가 개선은 별도 post-hoc experiment로 분리\n");
fprintf("- 최종 프로젝트 결론은 Recall만이 아니라 FPR/F1/AUROC를 함께 보고 판단\n\n");

fprintf("ChatGPT에 아래를 보내주세요.\n");
fprintf("1) FINAL RUN-LEVEL TEST COMPARISON\n");
fprintf("2) FALSE ALARMS\n");
fprintf("3) MISSED FAULTS\n");
fprintf("4) STEP 09 CHECK\n\n");

%% ================= LOCAL FUNCTIONS =================

function T = readScoreTable(file)
    opts = detectImportOptions(file,"VariableNamingRule","preserve");
    T = readtable(file,opts);

    if ismember("Run",string(T.Properties.VariableNames))
        T.Run = string(T.Run);
    end

    if ismember("Split",string(T.Properties.VariableNames))
        T.Split = string(T.Split);
    end
end

function y = safeDiv(a,b)
    if b==0
        y = NaN;
    else
        y = a/b;
    end
end
