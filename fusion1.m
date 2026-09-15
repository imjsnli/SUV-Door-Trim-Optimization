%% GBDT + Kriging 融合模型 + 对比模型（SSA-GBDT 及未优化 GBDT）
% 包含绘图、性能对比表格、Excel导出，自动保存所有图片（fig + png）
clear
clc
close all

% 创建图片保存文件夹（绝对路径，确保可写入）
fig_dir = fullfile(pwd, 'Figures');
if ~exist(fig_dir, 'dir')
    [status, msg] = mkdir(fig_dir);
    if ~status
        error('无法创建图片保存文件夹 %s: %s', fig_dir, msg);
    end
end

%% 1. 加载数据
fprintf('========== 加载数据 ==========\n');
res = xlsread('data.xlsx');
x = res(:, 2:11);
y = res(:, 12);

n_samples = size(x, 1);
n_features = size(x, 2);
fprintf('样本总数: %d\n', n_samples);
fprintf('特征数量: %d\n', n_features);

%% 2. 划分训练集/测试集（固定随机种子50）
fprintf('\n========== 划分数据集 ==========\n');
rng(50);   % 固定种子，保证可复现
train_ratio = 0.80;
n_train = floor(train_ratio * n_samples);

indices = randperm(n_samples);
train_idx = indices(1:n_train);
test_idx = indices(n_train+1:end);

X_train = x(train_idx, :);
Y_train = y(train_idx);
X_test = x(test_idx, :);
Y_test = y(test_idx);

Y_train = Y_train(:);
Y_test = Y_test(:);

fprintf('训练集样本数: %d\n', length(train_idx));
fprintf('测试集样本数: %d\n', length(test_idx));

%% ==================== PSO搜索GBDT最优参数 ====================
fprintf('\n========== PSO搜索GBDT最优参数（5折交叉验证） ==========\n');

% 参数边界（注意整数参数需取整）
% 顺序：[n_trees, learn_rate, min_leaf, max_splits]
lb = [20, 0.05, 2, 3];
ub = [60, 0.3, 10, 10];

% PSO参数
max_iter = 30;      % 迭代次数
pop_size = 20;      % 种群大小
w = 0.8;            % 惯性权重
c1 = 1.5;           % 个体学习因子
c2 = 1.5;           % 社会学习因子

% 初始化种群和速度
dim = 4;
pop = zeros(pop_size, dim);
V = zeros(pop_size, dim);
fitness = inf(pop_size, 1);

% 生成固定的5折索引（用于搜索，保证可复现）
rng(1, 'twister');
cv_indices_search = crossvalind('Kfold', size(X_train,1), 5);
n_folds = max(cv_indices_search);

% 定义适应度函数（GBDT的5折CV RMSE）
fitness_func = @(params) compute_gbdt_cv_rmse(params, X_train, Y_train, cv_indices_search, n_folds);

% 随机初始化种群
for i = 1:pop_size
    pop(i,:) = lb + (ub - lb).*rand(1, dim);
    % 对整数参数取整
    pop(i,1) = round(pop(i,1));   % n_trees
    pop(i,3) = round(pop(i,3));   % min_leaf
    pop(i,4) = round(pop(i,4));   % max_splits
    % 越界处理
    pop(i,:) = max(pop(i,:), lb);
    pop(i,:) = min(pop(i,:), ub);
    fitness(i) = fitness_func(pop(i,:));
end

% 个体最优和全局最优
pbest = pop;
pbest_fitness = fitness;
[gbest_fitness, gbest_idx] = min(fitness);
gbest = pop(gbest_idx, :);

% 记录历史最优
hist_best = zeros(max_iter, 1);
hist_best(1) = gbest_fitness;

fprintf('初始最优适应度（GBDT CV RMSE）: %.4f\n', gbest_fitness);

% PSO主循环
for iter = 1:max_iter
    for i = 1:pop_size
        % 更新速度
        V(i,:) = w*V(i,:) + c1*rand(1,dim).*(pbest(i,:)-pop(i,:)) + c2*rand(1,dim).*(gbest-pop(i,:));
        % 速度限幅（可选）
        Vmax = 0.3*(ub-lb);
        V(i,:) = max(V(i,:), -Vmax);
        V(i,:) = min(V(i,:), Vmax);
        % 更新位置
        pop(i,:) = pop(i,:) + V(i,:);
        % 边界处理
        pop(i,:) = max(pop(i,:), lb);
        pop(i,:) = min(pop(i,:), ub);
        % 整数参数取整
        pop(i,1) = round(pop(i,1));
        pop(i,3) = round(pop(i,3));
        pop(i,4) = round(pop(i,4));
        % 重新边界处理（取整后可能越界）
        pop(i,:) = max(pop(i,:), lb);
        pop(i,:) = min(pop(i,:), ub);
        % 计算适应度
        fitness(i) = fitness_func(pop(i,:));
        % 更新个体最优
        if fitness(i) < pbest_fitness(i)
            pbest(i,:) = pop(i,:);
            pbest_fitness(i) = fitness(i);
        end
        % 更新全局最优
        if fitness(i) < gbest_fitness
            gbest_fitness = fitness(i);
            gbest = pop(i,:);
        end
    end
    hist_best(iter) = gbest_fitness;
    % 打印每代最优
    fprintf('迭代 %2d/%d, 当前最优GBDT CV RMSE = %.4f\n', iter, max_iter, gbest_fitness);
end

% 输出最终结果
fprintf('\nPSO搜索完成！最优参数:\n');
fprintf('  n_trees = %d\n', gbest(1));
fprintf('  learn_rate = %.3f\n', gbest(2));
fprintf('  min_leaf = %d\n', gbest(3));
fprintf('  max_splits = %d\n', gbest(4));
fprintf('  最优GBDT交叉验证RMSE = %.4f\n', gbest_fitness);

% 将最优参数赋给后续变量
n_trees = gbest(1);
learn_rate = gbest(2);
min_leaf = gbest(3);
max_splits = gbest(4);

%% 3. 使用最优参数训练最终PSO-GBDT模型
fprintf('\n========== 使用最优参数训练最终PSO-GBDT模型 ==========\n');

tree_template = templateTree(...
    'MaxNumSplits', max_splits, ...
    'MinLeafSize', min_leaf);

tic;
gbdt_model1 = fitrensemble(...
    X_train, Y_train, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', n_trees, ...
    'Learners', tree_template, ...
    'LearnRate', learn_rate);
time_pso_gbdt = toc;
fprintf('PSO-GBDT训练完成，耗时: %.2f 秒\n', time_pso_gbdt);

Y_train_pso_gbdt = predict(gbdt_model1, X_train);
Y_test_pso_gbdt = predict(gbdt_model1, X_test);
Y_train_pso_gbdt = Y_train_pso_gbdt(:);
Y_test_pso_gbdt = Y_test_pso_gbdt(:);

rmse_train_pso_gbdt = sqrt(mean((Y_train_pso_gbdt - Y_train).^2));
sigma2_gbdt1 = rmse_train_pso_gbdt^2;
fprintf('PSO-GBDT 训练集RMSE: %.4f (方差: %.6f)\n', rmse_train_pso_gbdt, sigma2_gbdt1);

%% 4. Kriging模型训练
fprintf('\n========== 训练Kriging模型 ==========\n');

tic;
kriging_model1 = fitrgp(X_train, Y_train, ...
    'KernelFunction', 'ardmatern52', ...
    'BasisFunction', 'constant', ...
    'Standardize', true, ...
    'FitMethod', 'sd', ...
    'PredictMethod', 'sd');
time_kriging = toc;
fprintf('Kriging训练完成，耗时: %.2f 秒\n', time_kriging);

[Y_train_kriging, ysd_train, ~] = predict(kriging_model1, X_train);
[Y_test_kriging, ysd_test, ~] = predict(kriging_model1, X_test);

sigma2_train_kriging = ysd_train(:).^2;
sigma2_test_kriging = ysd_test(:).^2;
Y_train_kriging = Y_train_kriging(:);
Y_test_kriging = Y_test_kriging(:);

%% 5. 融合模型（基于PSO-GBDT与Kriging）
% ===== 可调整参数 =====
max_weight_gbdt1 = 0.5;      % GBDT权重上限
variance_floor1 = 1e-10;     % 方差下限
% =======================

fprintf('\n========== 构建融合模型（PSO-GBDT权重最大%.2f，无截断） ==========\n', max_weight_gbdt1);

sigma2_train_kriging = max(sigma2_train_kriging, variance_floor1);
sigma2_test_kriging = max(sigma2_test_kriging, variance_floor1);

ratio_train = sigma2_train_kriging ./ (sigma2_gbdt1 + sigma2_train_kriging);
ratio_test  = sigma2_test_kriging  ./ (sigma2_gbdt1 + sigma2_test_kriging);

w_gbdt_train = max_weight_gbdt1 * ratio_train;
w_kriging_train = 1 - w_gbdt_train;

w_gbdt_test = max_weight_gbdt1 * ratio_test;
w_kriging_test = 1 - w_gbdt_test;

Y_train_fusion = w_gbdt_train .* Y_train_pso_gbdt + w_kriging_train .* Y_train_kriging;
Y_test_fusion = w_gbdt_test .* Y_test_pso_gbdt + w_kriging_test .* Y_test_kriging;

%% ==================== Equal Fusion：等权融合 ====================
fprintf('\n========== 构建Equal Fusion（PSO-GBDT + Kriging） ==========\n');

Y_train_equal_fusion = ...
    0.5 .* Y_train_pso_gbdt + 0.5 .* Y_train_kriging;

Y_test_equal_fusion = ...
    0.5 .* Y_test_pso_gbdt + 0.5 .* Y_test_kriging;

fprintf('Equal Fusion构建完成：PSO-GBDT权重=0.5，Kriging权重=0.5\n');

fprintf('\n--- 融合权重统计（测试集）---\n');
fprintf('PSO-GBDT权重: 均值=%.4f, 最小=%.4f, 最大=%.4f\n', ...
    mean(w_gbdt_test), min(w_gbdt_test), max(w_gbdt_test));
fprintf('Kriging权重:   均值=%.4f, 最小=%.4f, 最大=%.4f\n', ...
    mean(w_kriging_test), min(w_kriging_test), max(w_kriging_test));

%% ==================== SSA-GBDT模型 ====================
fprintf('\n========== SSA搜索GBDT最优参数（5折交叉验证） ==========\n');

% SSA参数设置
N_ssa = 20;          % 种群大小
Max_iter_ssa = 30;   % 最大迭代次数

% 参数边界与PSO相同
lb_ssa = [20, 0.05, 2, 3];
ub_ssa = [60, 0.3, 10, 10];
dim_ssa = 4;

% 适应度函数（复用相同的cv_indices_search）
fitness_ssa_func = @(params) compute_gbdt_cv_rmse(params, X_train, Y_train, cv_indices_search, n_folds);

fprintf('SSA优化进行中...\n');
[best_fitness_ssa, best_params_ssa, conv_curve] = SSA_optimization(N_ssa, Max_iter_ssa, lb_ssa, ub_ssa, dim_ssa, fitness_ssa_func);

fprintf('SSA优化完成！最优适应度（CV RMSE）: %.4f\n', best_fitness_ssa);
% 对整数参数取整
best_params_ssa(1) = round(best_params_ssa(1));
best_params_ssa(3) = round(best_params_ssa(3));
best_params_ssa(4) = round(best_params_ssa(4));
% 边界处理
best_params_ssa = max(best_params_ssa, lb_ssa);
best_params_ssa = min(best_params_ssa, ub_ssa);
fprintf('最优参数: n_trees=%d, learn_rate=%.3f, min_leaf=%d, max_splits=%d\n', ...
    best_params_ssa(1), best_params_ssa(2), best_params_ssa(3), best_params_ssa(4));

% 使用最优参数训练SSA-GBDT模型
fprintf('\n========== 训练SSA-GBDT模型 ==========\n');
tree_template_ssa = templateTree(...
    'MaxNumSplits', best_params_ssa(4), ...
    'MinLeafSize', best_params_ssa(3));

tic;
ssa_gbdt_model = fitrensemble(...
    X_train, Y_train, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', best_params_ssa(1), ...
    'Learners', tree_template_ssa, ...
    'LearnRate', best_params_ssa(2));
time_ssa_gbdt = toc;
fprintf('SSA-GBDT训练完成，耗时: %.2f 秒\n', time_ssa_gbdt);

Y_train_ssa_gbdt = predict(ssa_gbdt_model, X_train);
Y_test_ssa_gbdt = predict(ssa_gbdt_model, X_test);
Y_train_ssa_gbdt = Y_train_ssa_gbdt(:);
Y_test_ssa_gbdt = Y_test_ssa_gbdt(:);

%% ==================== 新增：未优化的GBDT基线模型 ====================
fprintf('\n========== 训练未优化的GBDT基线模型（默认参数） ==========\n');
% 使用fitrensemble的默认参数：NumLearningCycles=100, LearnRate=0.1, 默认树（无剪枝限制）
tic;
default_gbdt_model = fitrensemble(X_train, Y_train, 'Method', 'LSBoost');
time_default_gbdt = toc;
fprintf('默认GBDT训练完成，耗时: %.2f 秒\n', time_default_gbdt);

Y_train_default_gbdt = predict(default_gbdt_model, X_train);
Y_test_default_gbdt = predict(default_gbdt_model, X_test);
Y_train_default_gbdt = Y_train_default_gbdt(:);
Y_test_default_gbdt = Y_test_default_gbdt(:);

% ==================== 新增结束 ====================

%% 6. 性能评估 —— 计算所有模型的训练集和测试集指标
fprintf('\n========== 性能评估（训练集 & 测试集） ==========\n');

calc_metrics = @(y_pred, y_true) struct(...
    'RMSE', sqrt(mean((y_pred - y_true).^2)), ...
    'R2', 1 - sum((y_true - y_pred).^2) / sum((y_true - mean(y_true)).^2));

% 模型顺序：PSO-GBDT, Kriging, Equal Fusion, 动态融合, SSA-GBDT, 默认GBDT
model_names = {'PSO-GBDT', 'Kriging', 'Equal Fusion', ...
    'PSO-GBDT+Kriging', 'SSA-GBDT', 'GBDT(默认)'};

train_preds = {Y_train_pso_gbdt, Y_train_kriging, ...
    Y_train_equal_fusion, Y_train_fusion, ...
    Y_train_ssa_gbdt, Y_train_default_gbdt};

test_preds = {Y_test_pso_gbdt, Y_test_kriging, ...
    Y_test_equal_fusion, Y_test_fusion, ...
    Y_test_ssa_gbdt, Y_test_default_gbdt};
metrics_table = zeros(length(model_names), 4);

for i = 1:length(model_names)
    if all(~isnan(train_preds{i}))
        train_metrics = calc_metrics(train_preds{i}, Y_train);
        test_metrics  = calc_metrics(test_preds{i}, Y_test);
    else
        train_metrics.RMSE = NaN; train_metrics.R2 = NaN;
        test_metrics.RMSE  = NaN; test_metrics.R2  = NaN;
    end
    metrics_table(i, :) = [train_metrics.RMSE, train_metrics.R2, test_metrics.RMSE, test_metrics.R2];
end

% 输出对比表格
fprintf('\n%-14s %-14s %-14s %-14s %-14s\n', ...
    '模型', '训练集RMSE', '训练集R²', '测试集RMSE', '测试集R²');
fprintf('%s\n', repmat('-', 1, 75));
for i = 1:length(model_names)
    fprintf('%-14s %-14.4f %-14.4f %-14.4f %-14.4f\n', ...
        model_names{i}, metrics_table(i,1), metrics_table(i,2), ...
        metrics_table(i,3), metrics_table(i,4));
end

%% 7. 特征重要性（PSO-GBDT）
fprintf('\n========== 特征重要性(PSO-GBDT) ==========\n');
try
    imp = predictorImportance(gbdt_model1);
    [sorted_imp, sort_idx] = sort(imp, 'descend');
    for i = 1:n_features
        fprintf('特征 %2d: 重要性 = %.4f\n', sort_idx(i), sorted_imp(i));
    end
catch
    fprintf('无法提取GBDT特征重要性。\n');
end

%% 8. 绘图
% 图1：测试集预测对比（所有模型）+ 指标标注
figure('Position', [100, 100, 800, 600]);
plot(1:length(Y_test), Y_test, 'k-o', 'MarkerSize', 6, 'LineWidth', 1.5); hold on;
plot(1:length(Y_test_pso_gbdt), Y_test_pso_gbdt, 'b-s', 'MarkerSize', 5, 'LineWidth', 1);
plot(1:length(Y_test_kriging), Y_test_kriging, 'r-^', 'MarkerSize', 5, 'LineWidth', 1);
plot(1:length(Y_test_equal_fusion), Y_test_equal_fusion, 'MarkerSize', 5, 'LineWidth', 1);
plot(1:length(Y_test_fusion), Y_test_fusion, 'g-d', 'MarkerSize', 5, 'LineWidth', 1);
plot(1:length(Y_test_ssa_gbdt), Y_test_ssa_gbdt, 'm-*', 'MarkerSize', 5, 'LineWidth', 1);
plot(1:length(Y_test_default_gbdt), Y_test_default_gbdt, 'c-x', 'MarkerSize', 5, 'LineWidth', 1);
xlabel('测试样本索引');
ylabel('预测值 / 真实值');
title('测试集预测对比');
legend('真实值', 'PSO-GBDT', 'Kriging', 'Equal Fusion', ...
    'PSO-GBDT+Kriging', 'SSA-GBDT', 'GBDT(默认)', ...
    'Location', 'best');
grid on;

% 添加测试集指标文本框（左上角）
str_test = sprintf([...
    'PSO-GBDT: RMSE=%.4f, R²=%.4f\n', ...
    'Kriging: RMSE=%.4f, R²=%.4f\n', ...
    'Equal Fusion: RMSE=%.4f, R²=%.4f\n', ...
    'PSO-GBDT+Kriging: RMSE=%.4f, R²=%.4f\n', ...
    'SSA-GBDT: RMSE=%.4f, R²=%.4f\n', ...
    'GBDT(默认): RMSE=%.4f, R²=%.4f'], ...
    metrics_table(1,3), metrics_table(1,4), ...
    metrics_table(2,3), metrics_table(2,4), ...
    metrics_table(3,3), metrics_table(3,4), ...
    metrics_table(4,3), metrics_table(4,4), ...
    metrics_table(5,3), metrics_table(5,4), ...
    metrics_table(6,3), metrics_table(6,4));
text(0.02, 0.98, str_test, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
    'BackgroundColor', 'white', 'EdgeColor', 'k', 'FontSize', 9);

% 保存图1（确保文件夹存在）
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
savefig(gcf, fullfile(fig_dir, 'Fig1_TestSetComparison.fig'));
exportgraphics(gcf, fullfile(fig_dir, 'Fig1_TestSetComparison.png'), 'Resolution', 300);

% 图2：训练集预测对比（所有模型）+ 指标标注
figure('Position', [150, 150, 800, 600]);
plot(1:length(Y_train), Y_train, 'k-o', 'MarkerSize', 4, 'LineWidth', 1.5); hold on;
plot(1:length(Y_train_pso_gbdt), Y_train_pso_gbdt, 'b-s', 'MarkerSize', 3, 'LineWidth', 1);
plot(1:length(Y_train_kriging), Y_train_kriging, 'r-^', 'MarkerSize', 3, 'LineWidth', 1);
plot(1:length(Y_train_equal_fusion), Y_train_equal_fusion, ...
    '--', 'MarkerSize', 3, 'LineWidth', 1);
plot(1:length(Y_train_fusion), Y_train_fusion, 'g-d', 'MarkerSize', 3, 'LineWidth', 1);
plot(1:length(Y_train_ssa_gbdt), Y_train_ssa_gbdt, 'm-*', 'MarkerSize', 3, 'LineWidth', 1);
plot(1:length(Y_train_default_gbdt), Y_train_default_gbdt, 'c-x', 'MarkerSize', 3, 'LineWidth', 1);
xlabel('训练样本索引');
ylabel('预测值 / 真实值');
title('训练集预测对比');
legend('真实值', 'PSO-GBDT', 'Kriging', 'Equal Fusion', ...
    'PSO-GBDT+Kriging', 'SSA-GBDT', 'GBDT(默认)', ...
    'Location', 'best');
grid on;

% 添加训练集指标文本框（左上角）
str_train = sprintf([...
    'PSO-GBDT: RMSE=%.4f, R²=%.4f\n', ...
    'Kriging: RMSE=%.4f, R²=%.4f\n', ...
    'Equal Fusion: RMSE=%.4f, R²=%.4f\n', ...
    'PSO-GBDT+Kriging: RMSE=%.4f, R²=%.4f\n', ...
    'SSA-GBDT: RMSE=%.4f, R²=%.4f\n', ...
    'GBDT(默认): RMSE=%.4f, R²=%.4f'], ...
    metrics_table(1,1), metrics_table(1,2), ...
    metrics_table(2,1), metrics_table(2,2), ...
    metrics_table(3,1), metrics_table(3,2), ...
    metrics_table(4,1), metrics_table(4,2), ...
    metrics_table(5,1), metrics_table(5,2), ...
    metrics_table(6,1), metrics_table(6,2));
text(0.02, 0.98, str_train, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
    'BackgroundColor', 'white', 'EdgeColor', 'k', 'FontSize', 9);

% 保存图2
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
savefig(gcf, fullfile(fig_dir, 'Fig2_TrainSetComparison.fig'));
exportgraphics(gcf, fullfile(fig_dir, 'Fig2_TrainSetComparison.png'), 'Resolution', 300);

% 图3：融合模型回归图（训练集+测试集）
figure('Position', [200, 200, 800, 700]);
scatter(Y_train, Y_train_fusion, 50, 'b', 'filled', 'DisplayName', '训练集'); hold on;
scatter(Y_test, Y_test_fusion, 50, 'r', 'filled', 'DisplayName', '测试集');
min_val = min([Y_train; Y_test; Y_train_fusion; Y_test_fusion]);
max_val = max([Y_train; Y_test; Y_train_fusion; Y_test_fusion]);
plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.5, 'DisplayName', 'y = x');
rmse_train = sqrt(mean((Y_train_fusion - Y_train).^2));
r2_train = 1 - sum((Y_train - Y_train_fusion).^2) / sum((Y_train - mean(Y_train)).^2);
rmse_test = sqrt(mean((Y_test_fusion - Y_test).^2));
r2_test = 1 - sum((Y_test - Y_test_fusion).^2) / sum((Y_test - mean(Y_test)).^2);
text_str = sprintf('训练集: RMSE=%.4f, R²=%.4f\n测试集: RMSE=%.4f, R²=%.4f', ...
    rmse_train, r2_train, rmse_test, r2_test);
x_pos = min_val + 0.05*(max_val - min_val);
y_pos = max_val - 0.15*(max_val - min_val);
text(x_pos, y_pos, text_str, 'FontSize', 11, 'BackgroundColor', 'white', ...
    'EdgeColor', 'k', 'VerticalAlignment', 'top');
xlabel('真实值'); ylabel('融合模型预测值');
title('融合模型（PSO-GBDT+Kriging）预测值与真实值散点图');
legend('Location', 'best'); grid on; axis equal;
xlim([min_val-0.1*(max_val-min_val), max_val+0.1*(max_val-min_val)]);
ylim([min_val-0.1*(max_val-min_val), max_val+0.1*(max_val-min_val)]);

% 保存图3
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
savefig(gcf, fullfile(fig_dir, 'Fig3_FusionRegression.fig'));
exportgraphics(gcf, fullfile(fig_dir, 'Fig3_FusionRegression.png'), 'Resolution', 300);

% 图4：所有模型训练集+测试集散点回归对比
fprintf('\n========== 绘制所有模型训练集+测试集散点回归对比图 ==========\n');
figure('Position', [100, 100, 1200, 800]);
num_models = length(model_names);
scatter_model_names = model_names;
scatter_model_names(strcmp(scatter_model_names, 'GBDT(默认)')) = {'GBDT'};
for i = 1:num_models
    subplot(2, 3, i);
    y_true_train = Y_train;
    y_pred_train = train_preds{i};
    y_true_test = Y_test;
    y_pred_test = test_preds{i};
    
    if any(isnan(y_pred_train)) || any(isnan(y_pred_test))
        text(0.5, 0.5, '预测失败', 'HorizontalAlignment', 'center', 'FontSize', 14);
        set(gca, 'XTick', [], 'YTick', []);
        title(scatter_model_names{i});
        continue;
    end
    
    % 绘制训练集散点（蓝色）
    scatter(y_true_train, y_pred_train, 30, 'b', 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Train');
    hold on;
    % 绘制测试集散点（红色）
    scatter(y_true_test, y_pred_test, 30, 'r', 'filled', 'MarkerFaceAlpha', 0.5, 'DisplayName', 'Test');
    
    % 计算坐标范围（包含所有点）
    all_vals = [y_true_train; y_pred_train; y_true_test; y_pred_test];
    min_val = min(all_vals);
    max_val = max(all_vals);
    margin = 0.1 * (max_val - min_val);
    if margin == 0, margin = 0.5; end
    plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin], 'k--', 'LineWidth', 1.5, 'DisplayName', 'y = x');
    xlabel('Actual value');
    ylabel('Predicted value');
    title(scatter_model_names{i});
    grid on;
    axis equal;
    xlim([min_val-margin, max_val+margin]);
    ylim([min_val-margin, max_val+margin]);
    
    % 计算训练集和测试集的RMSE和R²
    rmse_train = sqrt(mean((y_pred_train - y_true_train).^2));
    r2_train = 1 - sum((y_true_train - y_pred_train).^2) / sum((y_true_train - mean(y_true_train)).^2);
    rmse_test = sqrt(mean((y_pred_test - y_true_test).^2));
    r2_test = 1 - sum((y_true_test - y_pred_test).^2) / sum((y_true_test - mean(y_true_test)).^2);
    
    % 在左上角显示两个集合的指标
    text_str = sprintf('Train: RMSE=%.4f, R²=%.4f\nTest: RMSE=%.4f, R²=%.4f', ...
        rmse_train, r2_train, rmse_test, r2_test);
    text(0.05, 0.95, text_str, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
        'BackgroundColor', 'white', 'EdgeColor', 'k', 'FontSize', 9);
    
    % 添加图例（仅显示训练/测试，不包含y=x）
    legend('Train', 'Test', 'Location', 'southeast');
end
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

% 保存图4
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
savefig(gcf, fullfile(fig_dir, 'Fig4_AllModelsScatter.fig'));
exportgraphics(gcf, fullfile(fig_dir, 'Fig4_AllModelsScatter.png'), 'Resolution', 300);

%% 9. 保存模型
save('fusion_model_GBDT_PSOsearch_with_SVR1.mat', ...
    'gbdt_model1', 'kriging_model1', 'sigma2_gbdt1', ...
    'max_weight_gbdt1', 'variance_floor1');
fprintf('\n模型已保存至 fusion_model_GBDT_PSOsearch_with_SVR1.mat\n');

%% 10. 保存所有数据到Excel
fprintf('\n========== 保存数据到Excel ==========\n');
excel_file = 'ModelResults_PSOGBDT_Kriging_SSAGBDT_defaultGBDT.xlsx';

% 训练集数据
train_data = table(...
    Y_train, ...
    Y_train_pso_gbdt, ...
    Y_train_kriging, ...
    Y_train_equal_fusion, ...
    Y_train_fusion, ...
    Y_train_ssa_gbdt, ...
    Y_train_default_gbdt, ...
    'VariableNames', {...
    '真实值', ...
    'PSO_GBDT', ...
    'Kriging', ...
    'Equal_Fusion', ...
    'PSO_GBDT_Kriging动态融合', ...
    'SSA_GBDT', ...
    'GBDT_默认'});

writetable(train_data, excel_file, 'Sheet', '训练集预测');

test_data = table(...
    Y_test, ...
    Y_test_pso_gbdt, ...
    Y_test_kriging, ...
    Y_test_equal_fusion, ...
    Y_test_fusion, ...
    Y_test_ssa_gbdt, ...
    Y_test_default_gbdt, ...
    'VariableNames', {...
    '真实值', ...
    'PSO_GBDT', ...
    'Kriging', ...
    'Equal_Fusion', ...
    'PSO_GBDT_Kriging动态融合', ...
    'SSA_GBDT', ...
    'GBDT_默认'});

writetable(test_data, excel_file, 'Sheet', '测试集预测');
% 性能指标
metric_names = {'训练集RMSE', '训练集R²', '测试集RMSE', '测试集R²'};
metrics_cell = [model_names', num2cell(metrics_table)];
metrics_table_final = cell2table(metrics_cell, 'VariableNames', [{'模型'}, metric_names]);
writetable(metrics_table_final, excel_file, 'Sheet', '性能指标');

fprintf('所有数据已保存至 %s\n', excel_file);

fprintf('========== 全部完成 ==========\n');

%% ==================== 辅助函数 ====================
% 1. GBDT交叉验证RMSE（用于PSO和SSA搜索）
function rmse_cv = compute_gbdt_cv_rmse(params, X, Y, cv_indices, n_folds)
    n_trees = round(params(1));
    learn_rate = params(2);
    min_leaf = round(params(3));
    max_splits = round(params(4));
    
    n_trees = max(n_trees, 20);   n_trees = min(n_trees, 120);
    learn_rate = max(learn_rate, 0.05);   learn_rate = min(learn_rate, 0.4);
    min_leaf = max(min_leaf, 2);   min_leaf = min(min_leaf, 12);
    max_splits = max(max_splits, 3);   max_splits = min(max_splits, 12);
    
    fold_rmse = zeros(n_folds, 1);
    for fold = 1:n_folds
        val_idx = (cv_indices == fold);
        train_idx = ~val_idx;
        X_train_cv = X(train_idx, :);
        Y_train_cv = Y(train_idx);
        X_val_cv = X(val_idx, :);
        Y_val_cv = Y(val_idx);
        try
            tree_template = templateTree('MaxNumSplits', max_splits, 'MinLeafSize', min_leaf);
            gbdt_cv = fitrensemble(X_train_cv, Y_train_cv, ...
                'Method', 'LSBoost', ...
                'NumLearningCycles', n_trees, ...
                'Learners', tree_template, ...
                'LearnRate', learn_rate);
            Y_val_pred = predict(gbdt_cv, X_val_cv);
            fold_rmse(fold) = sqrt(mean((Y_val_pred - Y_val_cv).^2));
        catch
            fold_rmse(fold) = inf;
        end
    end
    rmse_cv = mean(fold_rmse);
end

% 2. SSA优化算法实现（通用）
function [fMin, bestX, Convergence_curve] = SSA_optimization(N, Max_iter, lb, ub, dim, fobj)
    X = initialization(N, dim, ub, lb);
    fitness = zeros(N, 1);
    for i = 1:N
        fitness(i) = fobj(X(i,:));
    end
    [fMin, bestIndex] = min(fitness);
    bestX = X(bestIndex, :);
    Convergence_curve = zeros(1, Max_iter);
    
    for t = 1:Max_iter
        [~, sortedIdx] = sort(fitness);
        pBest = X(sortedIdx(1), :);
        pWorst = X(sortedIdx(end), :);
        
        for i = 1:round(N * 0.2)
            R2 = rand();
            for j = 1:dim
                if R2 < 0.8
                    X(i,j) = X(i,j) * exp(-i / (rand * Max_iter));
                else
                    X(i,j) = X(i,j) + randn * 1;
                end
            end
        end
        for i = round(N*0.2)+1 : N
            for j = 1:dim
                if i > N/2
                    X(i,j) = rand * (ub(j) - lb(j)) + lb(j);
                else
                    A = rand(1, dim); A(A>0.5) = 1; A(A<=0.5) = -1;
                    X(i,j) = pBest(j) + abs(X(i,j) - pBest(j)) * (A(j) / (A*A' + eps));
                end
            end
        end
        for i = 1:round(N * 0.1)
            idx = randi(N);
            for j = 1:dim
                if fitness(idx) > fMin
                    X(idx,j) = pBest(j) + randn * abs(X(idx,j) - pBest(j));
                else
                    X(idx,j) = X(idx,j) + (rand - 0.5) * 1;
                end
            end
        end
        for i = 1:N
            for j = 1:dim
                if X(i,j) < lb(j), X(i,j) = lb(j); end
                if X(i,j) > ub(j), X(i,j) = ub(j); end
            end
        end
        for i = 1:N
            fitness(i) = fobj(X(i,:));
        end
        [minFit, minIdx] = min(fitness);
        if minFit < fMin
            fMin = minFit;
            bestX = X(minIdx, :);
        end
        Convergence_curve(t) = fMin;
    end
end

% 3. 初始化函数
function X = initialization(N, dim, ub, lb)
    X = rand(N, dim) .* (ub - lb) + lb;
end
