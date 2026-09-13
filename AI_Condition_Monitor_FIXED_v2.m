function AI_Condition_Monitor_FIXED_v2
% AI_CONDITION_MONITOR
% NEW PROJECT: MATLAB AI Etch Autoencoder
% STEP 14 - MATLAB AI Condition Monitor GUI
%
% 기능
% 1) Test Run 선택
% 2) 3-Sigma / Isolation Forest / Autoencoder Run-level risk 비교
% 3) Autoencoder Step4 / Step5 Window Risk timeline 표시
% 4) AE anomaly interval 표시
% 5) Top-5 sensor troubleshooting priority 표시
% 6) Ground Truth는 "evaluation only"로 별도 표시
%
% 중요
% - 이 App은 STEP09~13에서 고정된 결과를 시각화/운영화한 GUI
% - 모델 재학습 / threshold retuning을 수행하지 않음
% - Top sensor는 root cause가 아니라 inspection priority
%
% 실행 위치
%   프로젝트/app/AI_Condition_Monitor.m
%
% 실행
%   AI_Condition_Monitor_FIXED_v2

clc;

%% 0. Project path
appDir = fileparts(mfilename("fullpath"));
projectRoot = fileparts(appDir);

tableDir = fullfile(projectRoot,"results","tables");

comparisonFile = fullfile(tableDir,"31_final_test_model_comparison.csv");
predictionFile = fullfile(tableDir,"32_final_test_run_predictions.csv");
top5File       = fullfile(tableDir,"45_ae_test_run_top5_normalized_sensors.csv");
windowFile     = fullfile(tableDir,"47_ae_test_window_risk_normalized.csv");
intervalFile   = fullfile(tableDir,"48_ae_anomaly_intervals.csv");
summaryFile    = fullfile(tableDir,"57_ae_troubleshooting_run_summary.csv");

requiredFiles = [ ...
    string(comparisonFile), ...
    string(predictionFile), ...
    string(top5File), ...
    string(windowFile), ...
    string(intervalFile), ...
    string(summaryFile)];

for i = 1:numel(requiredFiles)
    if ~isfile(requiredFiles(i))
        uialert(uifigure, ...
            "필수 결과 파일이 없습니다:" + newline + requiredFiles(i), ...
            "Missing File");
        return;
    end
end

%% 1. Load
C = readT(comparisonFile);
P = readT(predictionFile);
T5 = readT(top5File);
W = readT(windowFile);
I = readT(intervalFile);
S = readT(summaryFile);

% Sort runs for dropdown
runList = sort(P.Run);

%% 2. Main UI
fig = uifigure( ...
    Name="MATLAB AI Etch Condition Monitor", ...
    Position=[100 70 1450 860]);

main = uigridlayout(fig,[5 4]);
main.RowHeight = {70,120,'1x','1x',35};
main.ColumnWidth = {250,'1x','1x',360};
main.Padding = [12 12 12 12];
main.RowSpacing = 10;
main.ColumnSpacing = 10;

%% 2-1. Header
titleLabel = uilabel(main, ...
    Text="MATLAB AI Etch Condition Monitor", ...
    FontSize=24, ...
    FontWeight="bold", ...
    HorizontalAlignment="left");
titleLabel.Layout.Row = 1;
titleLabel.Layout.Column = [1 3];

subtitleLabel = uilabel(main, ...
    Text="3σ  |  Isolation Forest  |  Autoencoder  |  Sensor/Time Troubleshooting", ...
    FontSize=13);
subtitleLabel.Layout.Row = 1;
subtitleLabel.Layout.Column = [1 3];
subtitleLabel.VerticalAlignment = "bottom";

runPanel = uipanel(main,Title="Run Selection");
runPanel.Layout.Row = 1;
runPanel.Layout.Column = 4;

rg = uigridlayout(runPanel,[1 2]);
rg.ColumnWidth = {90,'1x'};

uilabel(rg,Text="Test Run");
defaultRun = runList(1);
if any(runList=="l3141.txm")
    defaultRun = "l3141.txm";
end

runDropDown = uidropdown(rg, ...
    Items=cellstr(runList), ...
    Value=char(defaultRun));

%% 2-2. Status cards
card1 = uipanel(main,Title="Autoencoder Decision");
card1.Layout.Row = 2;
card1.Layout.Column = 1;

g1 = uigridlayout(card1,[2 1]);
g1.RowHeight = {35,30};
aeDecisionLabel = uilabel(g1, ...
    Text="-", ...
    FontSize=22, ...
    FontWeight="bold", ...
    HorizontalAlignment="center");
aeRiskLabel = uilabel(g1, ...
    Text="Risk Ratio: -", ...
    HorizontalAlignment="center");

card2 = uipanel(main,Title="Ground Truth — Evaluation Only");
card2.Layout.Row = 2;
card2.Layout.Column = 2;

g2 = uigridlayout(card2,[2 1]);
g2.RowHeight = {35,30};
truthLabel = uilabel(g2, ...
    Text="-", ...
    FontSize=20, ...
    FontWeight="bold", ...
    HorizontalAlignment="center");
faultLabel = uilabel(g2, ...
    Text="-", ...
    HorizontalAlignment="center");

card3 = uipanel(main,Title="Primary Troubleshooting Priority");
card3.Layout.Row = 2;
card3.Layout.Column = 3;

g3 = uigridlayout(card3,[3 1]);
g3.RowHeight = {28,25,25};
prioritySensorLabel = uilabel(g3, ...
    Text="-", ...
    FontSize=17, ...
    FontWeight="bold", ...
    HorizontalAlignment="center");
prioritySubsystemLabel = uilabel(g3, ...
    Text="-", ...
    HorizontalAlignment="center");
priorityRiskLabel = uilabel(g3, ...
    Text="-", ...
    HorizontalAlignment="center");

card4 = uipanel(main,Title="Primary Anomaly Interval");
card4.Layout.Row = 2;
card4.Layout.Column = 4;

g4 = uigridlayout(card4,[3 1]);
g4.RowHeight = {25,25,25};
intervalStepLabel = uilabel(g4,Text="-",HorizontalAlignment="center");
intervalTimeLabel = uilabel(g4,Text="-",HorizontalAlignment="center");
intervalRiskLabel = uilabel(g4,Text="-",HorizontalAlignment="center");

%% 2-3. Risk timeline
timelinePanel = uipanel(main,Title="Autoencoder Window Risk Timeline");
timelinePanel.Layout.Row = 3;
timelinePanel.Layout.Column = [1 3];

tg = uigridlayout(timelinePanel,[1 1]);
riskAx = uiaxes(tg);
riskAx.XLabel.String = "Normalized Process Time";
riskAx.YLabel.String = "AE Risk Ratio";
riskAx.Title.String = "Step 4 / Step 5";
grid(riskAx,"on");

%% 2-4. Model comparison per selected Run
modelPanel = uipanel(main,Title="Selected Run — Model Comparison");
modelPanel.Layout.Row = 3;
modelPanel.Layout.Column = 4;

mg = uigridlayout(modelPanel,[1 1]);
modelTable = uitable(mg);
modelTable.ColumnSortable = true;

%% 2-5. Sensor priorities
sensorPanel = uipanel(main,Title="Top-5 Sensor Inspection Priority");
sensorPanel.Layout.Row = 4;
sensorPanel.Layout.Column = [1 2];

sg = uigridlayout(sensorPanel,[1 1]);
sensorTable = uitable(sg);
sensorTable.ColumnSortable = true;

%% 2-6. Sensor risk chart
sensorChartPanel = uipanel(main,Title="Top-5 Sensor Risk");
sensorChartPanel.Layout.Row = 4;
sensorChartPanel.Layout.Column = 3;

scg = uigridlayout(sensorChartPanel,[1 1]);
sensorAx = uiaxes(scg);
sensorAx.XLabel.String = "Sensor";
sensorAx.YLabel.String = "Normalized Sensor Risk";
grid(sensorAx,"on");

%% 2-7. Overall benchmark
benchmarkPanel = uipanel(main,Title="Main Test Benchmark");
benchmarkPanel.Layout.Row = 4;
benchmarkPanel.Layout.Column = 4;

bg = uigridlayout(benchmarkPanel,[1 1]);
benchmarkTable = uitable(bg);
benchmarkTable.Data = C(:, ...
    ["Model","Recall","F1","FPR","AUROC"]);
benchmarkTable.ColumnSortable = true;
benchmarkTable.ColumnWidth = {105,55,55,55,65};

%% 2-8. Footer
footer = uilabel(main, ...
    Text="Inspection priority only — not a root-cause diagnosis. STEP09 frozen Main Test model.", ...
    HorizontalAlignment="center", ...
    FontAngle="italic");
footer.Layout.Row = 5;
footer.Layout.Column = [1 4];

%% 3. Callback
runDropDown.ValueChangedFcn = @(src,event) updateRun(string(src.Value));

%% 4. Initial display
updateRun(string(runDropDown.Value));

%% ================= NESTED CALLBACK =================
    function updateRun(runName)

        %% A. Run prediction row
        pr = P(P.Run==runName,:);

        if height(pr)~=1
            return;
        end

        if logical(pr.Pred_AE)
            aeDecisionLabel.Text = "ANOMALY";
        else
            aeDecisionLabel.Text = "NORMAL";
        end

        aeRiskLabel.Text = sprintf( ...
            "AE Risk Ratio: %.3f | threshold = 1.0",pr.Risk_AE);

        truthLabel.Text = pr.Status;

        if strcmpi(pr.Status,"Fault")
            faultLabel.Text = "Fault: " + pr.Fault_Name;
        else
            faultLabel.Text = "Normal reference run";
        end

        %% B. Model table
        modelNames = ["3-Sigma";"Isolation Forest";"Autoencoder"];
        decisions = [ ...
            string(logicalToWord(pr.Pred_3Sigma)); ...
            string(logicalToWord(pr.Pred_IF)); ...
            string(logicalToWord(pr.Pred_AE))];

        risks = [pr.Risk_3Sigma;pr.Risk_IF;pr.Risk_AE];

        modelTable.Data = table( ...
            modelNames,risks,decisions, ...
            'VariableNames',{'Model','Risk_Ratio','Decision'});

        %% C. Top-5 sensors
        tp = T5(T5.Run==runName,:);
        tp = sortrows(tp,"Rank");

        if ~isempty(tp)
            subsystem = strings(height(tp),1);
            guide = strings(height(tp),1);

            for k = 1:height(tp)
                [subsystem(k),guide(k)] = mapSensor(tp.Sensor_Name(k));
            end

            beyondWord = repmat("NO",height(tp),1);
            beyondWord(tp.Run_Sensor_Max_Risk>1) = "YES";

            displayPriority = table( ...
                tp.Rank, ...
                tp.Sensor_Name, ...
                subsystem, ...
                tp.Run_Sensor_Max_Risk, ...
                beyondWord, ...
                'VariableNames', ...
                {'Rank','Sensor','Subsystem','Risk_Ratio','Beyond_Normal'});

            sensorTable.Data = displayPriority;
            sensorTable.ColumnWidth = {45,120,165,80,90};

            top1 = tp(1,:);
            [topSubsystem,~] = mapSensor(top1.Sensor_Name);

            if logical(pr.Pred_AE)
                if any(tp.Run_Sensor_Max_Risk>1)
                    card3.Title = "Primary Troubleshooting Priority";
                    prioritySensorLabel.Text = "#1  " + top1.Sensor_Name;
                    prioritySubsystemLabel.Text = topSubsystem;
                    priorityRiskLabel.Text = sprintf( ...
                        "Sensor Risk Ratio: %.3f (> 1.0)", ...
                        top1.Run_Sensor_Max_Risk);
                else
                    % Multivariate AE는 전체 패턴으로 anomaly가 날 수 있으므로,
                    % 개별 sensor가 정상 reference max를 넘지 않은 경우
                    % "abnormal sensor"라고 단정하지 않는다.
                    card3.Title = "Highest Relative Sensor — No Sensor > Ref";
                    prioritySensorLabel.Text = top1.Sensor_Name;
                    prioritySubsystemLabel.Text = topSubsystem;
                    priorityRiskLabel.Text = sprintf( ...
                        "Highest Sensor Risk: %.3f (< 1.0)", ...
                        top1.Run_Sensor_Max_Risk);
                end
            else
                card3.Title = "No Troubleshooting Trigger";
                prioritySensorLabel.Text = "AE NORMAL";
                prioritySubsystemLabel.Text = "No anomaly interval";
                priorityRiskLabel.Text = "Sensor ranking shown for reference only";
            end

            % Sensor risk chart:
            % categorical 축의 이전 category가 남는 현상을 피하기 위해 숫자 x축 사용
            cla(sensorAx,"reset");
            x = 1:height(tp);
            bar(sensorAx,x,tp.Run_Sensor_Max_Risk);
            yline(sensorAx,1,"--","Normal Reference Max");

            sensorAx.XTick = x;
            sensorAx.XTickLabel = cellstr(tp.Sensor_Name);
            sensorAx.XTickLabelRotation = 35;
            sensorAx.XLim = [0.5 height(tp)+0.5];
            sensorAx.XLabel.String = "Sensor";
            sensorAx.YLabel.String = "Normalized Sensor Risk";
            sensorAx.Title.String = "Top-5 Normalized Sensor Risk";
            grid(sensorAx,"on");
        end

        %% D. Window risk timeline
        wr = W(W.Run==runName,:);
        wr = sortrows(wr,["Step","Window_Index"]);

        cla(riskAx,"reset");
        riskAx.XLabel.String = "Normalized Process Time";
        riskAx.YLabel.String = "AE Risk Ratio";
        riskAx.Title.String = "Step 4 / Step 5";
        grid(riskAx,"on");
        hold(riskAx,"on");

        for st = [4 5]
            ws = wr(wr.Step==st,:);

            if isempty(ws)
                continue;
            end

            centerTime = ...
                (ws.Start_Norm_Time + ws.End_Norm_Time)/2;

            plot(riskAx, ...
                centerTime, ...
                ws.AE_Risk_Ratio, ...
                "-o", ...
                DisplayName="Step " + string(st));
        end

        yline(riskAx,1,"--","AE Threshold", ...
            DisplayName="Threshold");

        riskAx.XLim = [0 1];
        riskAx.YLimMode = "auto";
        legend(riskAx,"Location","best");
        hold(riskAx,"off");

        %% E. Primary anomaly interval
        ir = I(I.Run==runName,:);

        if isempty(ir)
            intervalStepLabel.Text = ...
                "No AE anomaly interval";

            intervalTimeLabel.Text = ...
                "Risk stayed below threshold";

            intervalRiskLabel.Text = ...
                "Localization not activated";

        else
            ir = sortrows(ir, ...
                ["Max_AE_Risk_Ratio","Mean_AE_Risk_Ratio"], ...
                ["descend","descend"]);

            strongest = ir(1,:);

            intervalStepLabel.Text = sprintf( ...
                "Step %d | Windows %d-%d", ...
                strongest.Step, ...
                strongest.First_Window_Index, ...
                strongest.Last_Window_Index);

            intervalTimeLabel.Text = sprintf( ...
                "Normalized Time: %.3f – %.3f", ...
                strongest.Interval_Start_Norm_Time, ...
                strongest.Interval_End_Norm_Time);

            intervalRiskLabel.Text = sprintf( ...
                "Interval Max Risk: %.3f", ...
                strongest.Max_AE_Risk_Ratio);

            % timeline에 primary interval 표시
            hold(riskAx,"on");
            hStart = xline(riskAx, ...
                strongest.Interval_Start_Norm_Time, ...
                ":", ...
                "Start");
            hStart.HandleVisibility = "off";

            hEnd = xline(riskAx, ...
                strongest.Interval_End_Norm_Time, ...
                ":", ...
                "End");
            hEnd.HandleVisibility = "off";
            hold(riskAx,"off");
        end
    end
end

%% ================= LOCAL FUNCTIONS =================

function T = readT(file)

opts = detectImportOptions(file,"VariableNamingRule","preserve");
T = readtable(file,opts);

vars = string(T.Properties.VariableNames);

if ismember("Run",vars)
    T.Run = string(T.Run);
end

if ismember("Split",vars)
    T.Split = string(T.Split);
end

if ismember("Status",vars)
    T.Status = string(T.Status);
end

if ismember("Fault_Name",vars)
    T.Fault_Name = string(T.Fault_Name);
end

if ismember("Sensor_Name",vars)
    T.Sensor_Name = string(T.Sensor_Name);
end

if ismember("Model",vars)
    T.Model = string(T.Model);
end
end

function word = logicalToWord(x)
if logical(x)
    word = "ANOMALY";
else
    word = "NORMAL";
end
end

function [subsystem,guide] = mapSensor(sensorName)

s = string(sensorName);

if any(s==["BCl3 Flow","Cl2 Flow"])
    subsystem = "Gas Delivery / MFC";
    guide = "Check gas-flow setpoint, MFC response, supply path, and recipe consistency.";

elseif any(s==["Pressure","Vat Valve"])
    subsystem = "Chamber Pressure / Vacuum";
    guide = "Check chamber pressure control, throttle/vat valve response, and vacuum path.";

elseif any(s==["RF Btm Pwr","RF Btm Rfl Pwr","RF Tuner", ...
               "RF Load","RF Phase Err","RF Pwr","RF Impedance"])
    subsystem = "RF Bias / Matching";
    guide = "Check forward/reflected power, match/tuner/load behavior, and RF delivery consistency.";

elseif any(s==["TCP Tuner","TCP Phase Err","TCP Impedance", ...
               "TCP Top Pwr","TCP Rfl Pwr","TCP Load"])
    subsystem = "TCP Source / Matching";
    guide = "Check TCP source power, reflected power, impedance, tuner/load, and matching state.";

elseif s=="He Press"
    subsystem = "Backside He / Chuck";
    guide = "Check backside He pressure, chuck contact/sealing, and wafer thermal-control path.";

elseif s=="Endpt A"
    subsystem = "Endpoint Signal";
    guide = "Check endpoint signal behavior, optical/sensor path, and process-endpoint consistency.";

else
    subsystem = "Other / Cross-coupled";
    guide = "Review this sensor with adjacent process variables and recipe state.";
end
end
