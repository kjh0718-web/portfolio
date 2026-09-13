%% step08_autoencoder.m
% NEW PROJECT: MATLAB AI Etch Autoencoder
% STEP 08 - Autoencoder Deep Learning Anomaly Detection
%
% 목적
% 1) Train Normal window만 사용해 Step4 / Step5 Autoencoder 학습
% 2) 입력을 bottleneck latent space로 압축 후 다시 복원
% 3) Window anomaly score = Reconstruction RMSE
% 4) Validation Normal은 early stopping / best model selection / threshold에만 사용
% 5) Fault label은 학습, 모델 선택, threshold 설정에 사용하지 않음
% 6) Test score는 생성/저장하지만 Fault/Normal 성능은 아직 공개하지 않음
%
% Autoencoder 구조
%   Input(D)
%      ↓
%   FC 64 + ReLU
%      ↓
%   FC 16 + ReLU   <- bottleneck / latent representation
%      ↓
%   FC 64 + ReLU
%      ↓
%   FC D            <- reconstruction
%
% D:
%   Step4 = 189 features
%   Step5 = 190 features
%
% 입력
%   data/processed/10_step4_model_ready.csv
%   data/processed/11_step5_model_ready.csv
%
% 출력
%   results/tables/27_autoencoder_thresholds.csv
%   results/tables/28_autoencoder_window_scores.csv
%   results/tables/29_autoencoder_run_scores.csv
%   results/tables/30_autoencoder_train_validation_summary.csv
%   models/step08_autoencoder.mat
%
% 중요
% - Window는 일부 overlap하지만 STEP 03에서 Run-level split을 먼저 수행했으므로
%   같은 Run의 Window가 Train / Validation / Test에 동시에 존재하지 않음.
% - 576 windows를 576개의 독립 Run으로 해석하지 않음.
% - 최종 성능은 Run-level 기준을 중심으로 비교 예정.

clc;
clear;
close all;

fprintf('\n============================================================\n');
fprintf(' NEW MATLAB AI ETCH PROJECT | STEP 08 AUTOENCODER\n');
fprintf('============================================================\n\n');

%% 0. 설정
RANDOM_SEED = 20260810;

ENCODER_WIDTH = 64;
LATENT_DIM = 16;
DECODER_WIDTH = 64;

MAX_EPOCHS = 200;
MINI_BATCH_SIZE = 64;
INITIAL_LEARNING_RATE = 1e-3;
VALIDATION_PATIENCE = 20;

TARGET_STEPS = [4 5];

fprintf("Model                      : Dense Autoencoder\n");
fprintf("Training data              : Train Normal only\n");
fprintf("Encoder                    : D -> %d -> %d\n", ...
    ENCODER_WIDTH,LATENT_DIM);
fprintf("Decoder                    : %d -> %d -> D\n", ...
    LATENT_DIM,DECODER_WIDTH);
fprintf("Loss                       : MSE reconstruction loss\n");
fprintf("Window anomaly score       : Reconstruction RMSE\n");
fprintf("Execution environment      : CPU\n");
fprintf("Test metrics               : SEALED until final comparison\n\n");

%% 1. 경로
projectRoot  = fileparts(fileparts(mfilename("fullpath")));
processedDir = fullfile(projectRoot,"data","processed");
tableDir     = fullfile(projectRoot,"results","tables");
modelDir     = fullfile(projectRoot,"models");

step4File = fullfile(processedDir,"10_step4_model_ready.csv");
step5File = fullfile(processedDir,"11_step5_model_ready.csv");

assert(isfile(step4File),"10_step4_model_ready.csv 파일이 없습니다.");
assert(isfile(step5File),"11_step5_model_ready.csv 파일이 없습니다.");

%% 2. Metadata 정의
metaNames = [ ...
    "Run","Experiment","Split","Status","Fault_Name", ...
    "Step","Window_Index","Start_Grid","End_Grid", ...
    "Start_Norm_Time","End_Norm_Time"];

windowTables = cell(numel(TARGET_STEPS),1);
runStepTables = cell(numel(TARGET_STEPS),1);

thresholdRows = table();
summaryRows = table();

autoencoders = struct();

%% 3. Step별 Autoencoder
for ii = 1:numel(TARGET_STEPS)

    st = TARGET_STEPS(ii);

    if st==4
        inputFile = step4File;
    else
        inputFile = step5File;
    end

    opts = detectImportOptions(inputFile,"VariableNamingRule","preserve");
    D = readtable(inputFile,opts);

    D.Run = string(D.Run);
    D.Split = string(D.Split);
    D.Status = string(D.Status);
    D.Fault_Name = string(D.Fault_Name);

    actualVars = string(D.Properties.VariableNames);
    featureNames = actualVars(~ismember(actualVars,metaNames));

    X = double(D{:,cellstr(featureNames)});

    assert(all(isfinite(X),"all"), ...
        "Step %d 입력 데이터에 NaN/Inf가 있습니다.",st);

    trainMask = D.Split=="Train" & strcmpi(D.Status,"Normal");
    valMask   = D.Split=="Validation" & strcmpi(D.Status,"Normal");

    Xtrain = X(trainMask,:);
    Xval   = X(valMask,:);

    assert(size(Xtrain,1)==576, ...
        "Step %d Train window 수가 예상(576)과 다릅니다.",st);
    assert(size(Xval,1)==189, ...
        "Step %d Validation window 수가 예상(189)과 다릅니다.",st);

    numFeatures = size(Xtrain,2);

    fprintf("------------------------------------------------------------\n");
    fprintf("Step %d\n",st);
    fprintf("Train windows              : %d\n",size(Xtrain,1));
    fprintf("Validation windows         : %d\n",size(Xval,1));
    fprintf("Features                   : %d\n",numFeatures);
    fprintf("Train runs                 : %d\n", ...
        numel(unique(D.Run(trainMask))));
    fprintf("Validation runs            : %d\n", ...
        numel(unique(D.Run(valMask))));

    %% 3-1. Autoencoder architecture
    layers = [
        featureInputLayer(numFeatures, ...
            Normalization="none", ...
            Name="input")

        fullyConnectedLayer(ENCODER_WIDTH,Name="encoder_fc")
        reluLayer(Name="encoder_relu")

        fullyConnectedLayer(LATENT_DIM,Name="latent_fc")
        reluLayer(Name="latent_relu")

        fullyConnectedLayer(DECODER_WIDTH,Name="decoder_fc")
        reluLayer(Name="decoder_relu")

        fullyConnectedLayer(numFeatures,Name="reconstruction")
    ];

    %% 3-2. Training options
    % 576 / 64 = 9 mini-batches per epoch
    validationFrequency = ceil(size(Xtrain,1)/MINI_BATCH_SIZE);

    rng(RANDOM_SEED + st,"twister");

    options = trainingOptions("adam", ...
        InitialLearnRate=INITIAL_LEARNING_RATE, ...
        MaxEpochs=MAX_EPOCHS, ...
        MiniBatchSize=MINI_BATCH_SIZE, ...
        Shuffle="every-epoch", ...
        ValidationData={single(Xval),single(Xval)}, ...
        ValidationFrequency=validationFrequency, ...
        ValidationPatience=VALIDATION_PATIENCE, ...
        OutputNetwork="best-validation", ...
        ExecutionEnvironment="cpu", ...
        Plots="none", ...
        Verbose=false);

    %% 3-3. Train Normal only
    fprintf("Training Autoencoder...\n");

    [net,trainingInfo] = trainnet( ...
        single(Xtrain), ...
        single(Xtrain), ...
        layers, ...
        "mse", ...
        options);

    fprintf("Training complete.\n");

    %% 3-4. Reconstruction for all windows
    Xhat = minibatchpredict( ...
        net, ...
        single(X), ...
        MiniBatchSize=MINI_BATCH_SIZE);

    % 출력 자료형/방향 안전 처리
    if isa(Xhat,"dlarray")
        Xhat = extractdata(Xhat);
    end

    Xhat = double(Xhat);

    if size(Xhat,1) ~= size(X,1) && ...
            size(Xhat,2)==size(X,1) && ...
            size(Xhat,1)==size(X,2)
        Xhat = Xhat';
    end

    assert(isequal(size(Xhat),size(X)), ...
        "Step %d reconstruction size가 입력과 다릅니다.",st);

    residual = X - Xhat;

    % Window-level scalar anomaly score
    reconRMSE = sqrt(mean(residual.^2,2));

    assert(all(isfinite(reconRMSE)), ...
        "Step %d reconstruction score에 NaN/Inf가 있습니다.",st);

    trainScores = reconRMSE(trainMask);
    valScores   = reconRMSE(valMask);

    %% 3-5. Threshold calibration
    % Train/Validation은 Normal only.
    % 현재 알고 있는 정상 reconstruction error의 최댓값을 넘는 경우만 anomaly.
    trainMaxScore = max(trainScores);
    validationMaxScore = max(valScores);

    frozenThreshold = max(trainMaxScore,validationMaxScore);

    anomalyAll = reconRMSE > frozenThreshold;

    fprintf("Train RMSE mean            : %.6f\n",mean(trainScores));
    fprintf("Train RMSE max             : %.6f\n",trainMaxScore);
    fprintf("Validation RMSE mean       : %.6f\n",mean(valScores));
    fprintf("Validation RMSE max        : %.6f\n",validationMaxScore);
    fprintf("Frozen threshold           : %.6f\n",frozenThreshold);

    %% 3-6. Window score table
    % Test 정답 label은 score table에 넣지 않음
    W = table( ...
        D.Run(:), ...
        D.Experiment(:), ...
        D.Split(:), ...
        D.Step(:), ...
        D.Window_Index(:), ...
        D.Start_Norm_Time(:), ...
        D.End_Norm_Time(:), ...
        reconRMSE(:), ...
        anomalyAll(:), ...
        'VariableNames', ...
        {'Run','Experiment','Split','Step','Window_Index', ...
         'Start_Norm_Time','End_Norm_Time', ...
         'AE_Reconstruction_RMSE','Anomaly_AE'});

    windowTables{ii} = W;

    %% 3-7. Run-Step aggregation
    uniqueRuns = unique(W.Run,"stable");
    nRuns = numel(uniqueRuns);

    Run = strings(nRuns,1);
    Experiment = zeros(nRuns,1);
    Split = strings(nRuns,1);
    Step = repmat(st,nRuns,1);

    Window_Count = zeros(nRuns,1);
    Max_AE_RMSE = zeros(nRuns,1);
    Mean_AE_RMSE = zeros(nRuns,1);
    Threshold = repmat(frozenThreshold,nRuns,1);
    MaxScore_to_Threshold = zeros(nRuns,1);

    Anomalous_Window_Count = zeros(nRuns,1);
    Anomalous_Window_Fraction = zeros(nRuns,1);
    Run_Anomaly_AE = false(nRuns,1);

    for r = 1:nRuns

        runName = uniqueRuns(r);
        m = W.Run==runName;
        Wr = W(m,:);

        Run(r) = runName;
        Experiment(r) = Wr.Experiment(1);
        Split(r) = Wr.Split(1);
        Window_Count(r) = height(Wr);

        Max_AE_RMSE(r) = max(Wr.AE_Reconstruction_RMSE);
        Mean_AE_RMSE(r) = mean(Wr.AE_Reconstruction_RMSE);

        MaxScore_to_Threshold(r) = ...
            Max_AE_RMSE(r) / frozenThreshold;

        Anomalous_Window_Count(r) = sum(Wr.Anomaly_AE);

        Anomalous_Window_Fraction(r) = ...
            Anomalous_Window_Count(r) / Window_Count(r);

        Run_Anomaly_AE(r) = any(Wr.Anomaly_AE);
    end

    RS = table( ...
        Run,Experiment,Split,Step,Window_Count, ...
        Max_AE_RMSE,Mean_AE_RMSE,Threshold, ...
        MaxScore_to_Threshold, ...
        Anomalous_Window_Count, ...
        Anomalous_Window_Fraction, ...
        Run_Anomaly_AE);

    runStepTables{ii} = RS;

    %% 3-8. Threshold 기록
    tr = table( ...
        st, ...
        size(Xtrain,1), ...
        numel(unique(D.Run(trainMask))), ...
        numFeatures, ...
        ENCODER_WIDTH, ...
        LATENT_DIM, ...
        DECODER_WIDTH, ...
        MAX_EPOCHS, ...
        MINI_BATCH_SIZE, ...
        trainMaxScore, ...
        validationMaxScore, ...
        frozenThreshold, ...
        'VariableNames', ...
        {'Step','Train_Windows','Train_Runs','Feature_Count', ...
         'Encoder_Width','Latent_Dim','Decoder_Width', ...
         'Max_Epochs','Mini_Batch_Size', ...
         'Train_Normal_Max_RMSE', ...
         'Validation_Normal_Max_RMSE', ...
         'Frozen_Threshold'});

    thresholdRows = [thresholdRows;tr]; %#ok<AGROW>

    %% 3-9. Train / Validation summary only
    for splitName = ["Train","Validation"]

        wm = W.Split==splitName;
        rm = RS.Split==splitName;

        row = table( ...
            splitName,st, ...
            sum(wm), ...
            mean(W.AE_Reconstruction_RMSE(wm)), ...
            median(W.AE_Reconstruction_RMSE(wm)), ...
            max(W.AE_Reconstruction_RMSE(wm)), ...
            sum(W.Anomaly_AE(wm)), ...
            mean(W.Anomaly_AE(wm)), ...
            sum(rm), ...
            sum(RS.Run_Anomaly_AE(rm)), ...
            mean(RS.Run_Anomaly_AE(rm)), ...
            median(RS.Max_AE_RMSE(rm)), ...
            max(RS.Max_AE_RMSE(rm)), ...
            'VariableNames', ...
            {'Split','Step','Window_Count','Mean_Window_RMSE', ...
             'Median_Window_RMSE','Max_Window_RMSE', ...
             'Flagged_Windows','Window_FalseAlarm_Rate', ...
             'Run_Count','Flagged_Runs','Run_FalseAlarm_Rate', ...
             'Median_Run_MaxRMSE','Max_Run_RMSE'});

        summaryRows = [summaryRows;row]; %#ok<AGROW>
    end

    %% 3-10. Model 저장용 struct
    fieldName = sprintf("Step%d",st);

    autoencoders.(fieldName).Network = net;
    autoencoders.(fieldName).TrainingInfo = trainingInfo;
    autoencoders.(fieldName).FeatureNames = featureNames;
    autoencoders.(fieldName).FrozenThreshold = frozenThreshold;

    autoencoders.(fieldName).Architecture = struct( ...
        "InputDim",numFeatures, ...
        "EncoderWidth",ENCODER_WIDTH, ...
        "LatentDim",LATENT_DIM, ...
        "DecoderWidth",DECODER_WIDTH);

    autoencoders.(fieldName).Training = struct( ...
        "RandomSeed",RANDOM_SEED+st, ...
        "MaxEpochs",MAX_EPOCHS, ...
        "MiniBatchSize",MINI_BATCH_SIZE, ...
        "InitialLearningRate",INITIAL_LEARNING_RATE, ...
        "ValidationPatience",VALIDATION_PATIENCE, ...
        "TrainingSource","Train Normal only", ...
        "ValidationSource","Validation Normal only", ...
        "ExecutionEnvironment","cpu");

    fprintf("Train flagged windows      : %d / %d\n", ...
        sum(anomalyAll(trainMask)),sum(trainMask));

    fprintf("Validation flagged windows : %d / %d\n\n", ...
        sum(anomalyAll(valMask)),sum(valMask));
end

%% 4. Step4 + Step5 score 결합
windowScores = vertcat(windowTables{:});
runStepScores = vertcat(runStepTables{:});

allRuns = unique(runStepScores.Run,"stable");
nAllRuns = numel(allRuns);

Run = strings(nAllRuns,1);
Experiment = zeros(nAllRuns,1);
Split = strings(nAllRuns,1);

Step4_MaxRMSE = nan(nAllRuns,1);
Step5_MaxRMSE = nan(nAllRuns,1);

Step4_Threshold = nan(nAllRuns,1);
Step5_Threshold = nan(nAllRuns,1);

Step4_RiskRatio = nan(nAllRuns,1);
Step5_RiskRatio = nan(nAllRuns,1);

Combined_RiskRatio = zeros(nAllRuns,1);
Top_Step = zeros(nAllRuns,1);

Total_AnomalousWindows = zeros(nAllRuns,1);
Total_Windows = zeros(nAllRuns,1);
Anomalous_Window_Fraction = zeros(nAllRuns,1);

Run_Anomaly_AE = false(nAllRuns,1);

for r = 1:nAllRuns

    runName = allRuns(r);
    Rr = runStepScores(runStepScores.Run==runName,:);

    assert(height(Rr)==2, ...
        "Run %s에 Step4/Step5 AE score가 모두 존재하지 않습니다.",runName);

    s4 = Rr(Rr.Step==4,:);
    s5 = Rr(Rr.Step==5,:);

    Run(r) = runName;
    Experiment(r) = Rr.Experiment(1);
    Split(r) = Rr.Split(1);

    Step4_MaxRMSE(r) = s4.Max_AE_RMSE;
    Step5_MaxRMSE(r) = s5.Max_AE_RMSE;

    Step4_Threshold(r) = s4.Threshold;
    Step5_Threshold(r) = s5.Threshold;

    Step4_RiskRatio(r) = s4.MaxScore_to_Threshold;
    Step5_RiskRatio(r) = s5.MaxScore_to_Threshold;

    [Combined_RiskRatio(r),idx] = max( ...
        [Step4_RiskRatio(r),Step5_RiskRatio(r)]);

    if idx==1
        Top_Step(r) = 4;
    else
        Top_Step(r) = 5;
    end

    Total_AnomalousWindows(r) = ...
        s4.Anomalous_Window_Count + ...
        s5.Anomalous_Window_Count;

    Total_Windows(r) = ...
        s4.Window_Count + s5.Window_Count;

    Anomalous_Window_Fraction(r) = ...
        Total_AnomalousWindows(r) / Total_Windows(r);

    Run_Anomaly_AE(r) = ...
        s4.Run_Anomaly_AE || s5.Run_Anomaly_AE;
end

runScores = table( ...
    Run,Experiment,Split, ...
    Step4_MaxRMSE,Step4_Threshold,Step4_RiskRatio, ...
    Step5_MaxRMSE,Step5_Threshold,Step5_RiskRatio, ...
    Combined_RiskRatio,Top_Step, ...
    Total_AnomalousWindows,Total_Windows, ...
    Anomalous_Window_Fraction,Run_Anomaly_AE);

%% 5. Combined Train / Validation summary
for splitName = ["Train","Validation"]

    R = runScores(runScores.Split==splitName,:);

    row = table( ...
        splitName,0, ...
        NaN,NaN,NaN,NaN,NaN,NaN, ...
        height(R), ...
        sum(R.Run_Anomaly_AE), ...
        mean(R.Run_Anomaly_AE), ...
        median(R.Combined_RiskRatio), ...
        max(R.Combined_RiskRatio), ...
        'VariableNames', ...
        {'Split','Step','Window_Count','Mean_Window_RMSE', ...
         'Median_Window_RMSE','Max_Window_RMSE', ...
         'Flagged_Windows','Window_FalseAlarm_Rate', ...
         'Run_Count','Flagged_Runs','Run_FalseAlarm_Rate', ...
         'Median_Run_MaxRMSE','Max_Run_RMSE'});

    summaryRows = [summaryRows;row]; %#ok<AGROW>
end

%% 6. 저장
writetable(thresholdRows, ...
    fullfile(tableDir,"27_autoencoder_thresholds.csv"));

writetable(windowScores, ...
    fullfile(tableDir,"28_autoencoder_window_scores.csv"));

writetable(runScores, ...
    fullfile(tableDir,"29_autoencoder_run_scores.csv"));

writetable(summaryRows, ...
    fullfile(tableDir,"30_autoencoder_train_validation_summary.csv"));

save(fullfile(modelDir,"step08_autoencoder.mat"), ...
    "autoencoders", ...
    "RANDOM_SEED", ...
    "ENCODER_WIDTH", ...
    "LATENT_DIM", ...
    "DECODER_WIDTH", ...
    "MAX_EPOCHS", ...
    "MINI_BATCH_SIZE", ...
    "INITIAL_LEARNING_RATE", ...
    "VALIDATION_PATIENCE");

%% 7. 출력
fprintf("=== AUTOENCODER THRESHOLDS ===\n");
disp(thresholdRows);

fprintf("=== TRAIN / VALIDATION AUTOENCODER SUMMARY ===\n");
disp(summaryRows);

fprintf("\n============================================================\n");
fprintf(" STEP 08 CHECK\n");
fprintf("============================================================\n");

fprintf("Networks trained with Normal only      : true\n");
fprintf("Fault labels used for training?        : false\n");
fprintf("Fault labels used for threshold?       : false\n");
fprintf("Validation used for best model?        : true (Normal only)\n");
fprintf("Test performance opened?               : false\n");

fprintf("Window score rows                      : %d\n", ...
    height(windowScores));

fprintf("Run score rows                         : %d\n", ...
    height(runScores));

fprintf("All 127 Runs scored                    : %s\n", ...
    string(height(runScores)==127));

trainVal = runScores.Split=="Train" | ...
           runScores.Split=="Validation";

fprintf("Train/Validation Run false alarms      : %d / %d\n", ...
    sum(runScores.Run_Anomaly_AE(trainVal)),sum(trainVal));

fprintf("No Test metric in summary table        : %s\n", ...
    string(~any(summaryRows.Split=="Test")));

generated = [ ...
    isfile(fullfile(tableDir,"27_autoencoder_thresholds.csv")), ...
    isfile(fullfile(tableDir,"28_autoencoder_window_scores.csv")), ...
    isfile(fullfile(tableDir,"29_autoencoder_run_scores.csv")), ...
    isfile(fullfile(tableDir,"30_autoencoder_train_validation_summary.csv")), ...
    isfile(fullfile(modelDir,"step08_autoencoder.mat"))];

fprintf("All output files generated             : %s\n", ...
    string(all(generated)));

fprintf("\n============================================================\n");
fprintf(" STEP 08 COMPLETE\n");
fprintf("============================================================\n");

fprintf("\n중요 해석:\n");
fprintf("- Autoencoder는 정상 Window의 multivariate pattern을 압축/복원하도록 학습\n");
fprintf("- Reconstruction RMSE가 큰 Window를 비정상 후보로 판단\n");
fprintf("- Validation은 Normal만 사용하여 best network와 threshold 결정\n");
fprintf("- Fault/Test label은 어떤 tuning에도 사용하지 않음\n");
fprintf("- 다음 단계에서 3sigma / IF / AE를 동일 Test 42 Runs에서 최초 공개 비교\n\n");

fprintf("ChatGPT에 아래를 보내주세요.\n");
fprintf("1) AUTOENCODER THRESHOLDS\n");
fprintf("2) TRAIN / VALIDATION AUTOENCODER SUMMARY\n");
fprintf("3) STEP 08 CHECK\n\n");
