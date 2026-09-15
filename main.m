
clc;
clear;
close all;
warning('off', 'all') %关闭警告


%随机种子
rng(1)


%% Problem Definition

CostFunction=@(x) MOP(x);      % Cost Function

nVar=10;             % Number of Decision Variables

VarSize=[1 nVar];   % Size of Decision Variables Matrix

VarMin=ones(1,10);          % Lower Bound of Variables
VarMax=[9, 3, 7, 10, 8, 8, 6, 8, 3, 6];          % Upper Bound of Variables

% Number of Objective Functions
nObj=numel(CostFunction(unifrnd(VarMin,VarMax,VarSize)));


%% NSGA-III Parameters

MaxIt=100;      % Maximum Number of Iterations

nPop=100;        % Population Size

pCrossover=0.8;                         % Crossover Percentage
nCrossover=2*round(pCrossover*nPop/2);  % Number of Parnets (Offsprings)

pMutation=0.05;                          % Mutation Percentage
nMutation=round(pMutation*nPop);        % Number of Mutants

mu=0.01;                    % Mutation Rate

sigma=0.1*(VarMax-VarMin);  % Mutation Step Size


%% Initialization

empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.G=[];

pop=repmat(empty_individual,nPop,1);
M = 1000 ;  %惩罚系数
for i=1:nPop
    
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
    pop(i).G=Violation(pop(i).Position);    %约束违反   
    pop(i).Cost=CostFunction(pop(i).Position)+  M* pop(i).G;
end
[Z,nPop] = UniformPoint(nPop,nObj); %获取参考点
Zmin     = min([pop.Cost]',[],1);   %理想点
%% NSGA-III Main Loop


for it=1:MaxIt
    
    % Crossover
    popc=repmat(empty_individual,nCrossover/2,2);
    for k=1:nCrossover/2
        
        i1=randi([1 nPop]);
        p1=pop(i1);
        
        i2=randi([1 nPop]);
        p2=pop(i2);
        
        [popc(k,1).Position, popc(k,2).Position]=Crossover(p1.Position,p2.Position);
        
        % 添加边界检查
        newPos1 = max(min(popc(k,1).Position, VarMax), VarMin); % 确保新位置在边界内
        newPos2 = max(min(popc(k,2).Position, VarMax), VarMin);
        popc(k,1).Position = newPos1;
        popc(k,2).Position = newPos2;
        
        popc(k,1).G=Violation(popc(k,1).Position);    %约束违反   
        popc(k,1).Cost=CostFunction(popc(k,1).Position)+  M* popc(k,1).G;
        popc(k,2).G=Violation(popc(k,2).Position);    %约束违反   
        popc(k,2).Cost=CostFunction(popc(k,2).Position)+  M* popc(k,2).G;  
     
    end
    popc=popc(:);
    
    % Mutation
    popm=repmat(empty_individual,nMutation,1);
    for k=1:nMutation
        
        i=randi([1 nPop]);
        p=pop(i);
        
        popm(k).Position=Mutate(p.Position,mu,sigma);       
        newPos = max(min(popm(k).Position, VarMax), VarMin); % 确保新位置在边界内    
        popm(k).Position = newPos;
        popm(k).G = Violation(newPos);
        popm(k).Cost = CostFunction(newPos)+ M* popm(k).G;     
       
    end
    
    % Merge
    pop=[pop
         popc
         popm]; %#ok
     
     
    Zmin       = min([Zmin;[pop.Cost]'],[],1);
    [pop] = EnvironmentalSelection(pop,nPop,Z,Zmin);
    % Non-Dominated Sorting
    
    % Store F1
    [FrontNo,MaxFNo] = NDSort([pop.Cost]',nPop);
    F1=pop(FrontNo==1);
    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Number of F1 Members = ' num2str(numel(F1))]);
    
    % Plot F1 Costs
    figure(1);
    PlotCosts(F1);
    pause(0.01);
    
end


%保存pareto前沿图
savefig('pareto.fig');


%保存pareto解
pareto_set = reshape([F1.Position],[nVar,numel(F1)])';
pareto_front = [F1.Cost]';      %提取目标值 
pareto_front(:,1:3) = -pareto_front(:,1:3);  %将最大化的负值转为正值
pareto = [round(pareto_set),pareto_front];


%保存pareto数据
csvwrite('pareto.csv', pareto)

topsis



