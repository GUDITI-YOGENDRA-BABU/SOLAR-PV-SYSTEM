
%% ================= STANDALONE PV =================
clc; clear; close all;

%% === Load Dataset (STANDALONE) ===
genData = readtable('C:\Users\gudit\PVsyst7.0_Data\UserHourly\standalone without shanding_Project_VC0_HourlyRes_1.CSV');

genData.date = datetime(genData.date,'InputFormat','dd-MM-yyyy HH:mm:ss');

% Standalone condition: PV available
genData = genData(genData.GlobInc > 0, :);
disp(genData)

%% === Features & Target ===
X = table2array(genData(:, ...
    {'GlobHor','WindVel','TArray','T_Amb','GlobInc'}));

y = genData.E_Load;   % Power supplied to load

%% === STANDARDIZATION (VERY IMPORTANT) ===
X = normalize(X);

%% === Train-Test Split (80% / 20%) ===
cv = cvpartition(height(genData),'HoldOut',0.2);
XTrain = X(training(cv),:);  yTrain = y(training(cv));
XTest  = X(test(cv),:);      yTest  = y(test(cv));

%% === ML Models ===
models = {
    'Regression Tree',  @() fitrtree(XTrain,yTrain);
    'Random Forest',    @() fitrensemble(XTrain,yTrain,'Method','Bag');
    'SVR',              @() fitrsvm(XTrain,yTrain,'Standardize',true);
    'Gaussian Process', @() fitrgp(XTrain,yTrain,'Standardize',true);
    'Neural Network',   @() fitrnet(XTrain,yTrain);
};

numModels = size(models,1);
metrics = zeros(numModels,4);
yPreds  = zeros(length(yTest),numModels);

%% === Train & Evaluate ===
for i = 1:numModels
    modelName = models{i,1};
    fprintf('\nTraining Model: %s\n', modelName);

    model = models{i,2}();
    yPred = predict(model,XTest);

    rmse = sqrt(mean((yTest-yPred).^2));
    mae  = mean(abs(yTest-yPred));
    r2   = 1 - sum((yTest-yPred).^2)/sum((yTest-mean(yTest)).^2);

    % ---- SAFE MAPE (Standalone Fix) ----
    mape = mean(abs((yTest - yPred) ./ max(yTest,1))) * 100;

    metrics(i,:) = [rmse mae r2 mape];
    yPreds(:,i)  = yPred;

    fprintf('%s → RMSE: %.2f | MAE: %.2f | R²: %.4f | MAPE: %.2f %%\n', ...
        modelName, rmse, mae, r2, mape);
end

%% === Plot 1: Actual vs Predicted (Test Set) ===
N = min(100,length(yTest));
figure;
plot(yTest(1:N),'k','LineWidth',1.5); hold on;

colors = lines(numModels);
for i = 1:numModels
    plot(yPreds(1:N,i),'--','Color',colors(i,:),'LineWidth',1.2);
end

legend(['Actual', models(:,1)'], 'Location','northwest');
xlabel('Sample Index');
ylabel('Load Power (W)');
title('Standalone PV: Actual vs Predicted Load Power');
grid on;

%% === Plot 2: Model Performance Comparison ===
figure;
metricNames = {'RMSE','MAE','R2 Score','MAPE'};

for j = 1:4
    subplot(2,2,j);
    bar(metrics(:,j));
    set(gca,'XTickLabel',models(:,1),'XTickLabelRotation',30);
    ylabel(metricNames{j});
    title(metricNames{j});
    grid on;
end

sgtitle('Standalone PV: ML Model Performance Comparison');

