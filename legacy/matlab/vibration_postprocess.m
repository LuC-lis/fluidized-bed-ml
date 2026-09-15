% 主文件夹路径
main_folder = 'D:\desktop\zhendong';
output_main_folder = fullfile(main_folder, 'result'); % 统一结果保存路径

% 如果主结果文件夹不存在则创建
if ~exist(output_main_folder, 'dir')
    mkdir(output_main_folder);
end

% 配色方案
line_colors = {[0.8 0 0], [0.2 0.6 0.2], [0.2 0.2 0.6], [0.9 0.4 0]}; % 红、绿、蓝、橙

% 获取所有子文件夹（只包含文件夹，排除文件）
sub_folders = dir(main_folder);
sub_folders = sub_folders([sub_folders.isdir]); % 只保留文件夹
sub_folders = sub_folders(~ismember({sub_folders.name}, {'.', '..', 'result'})); % 排除特殊目录

% 循环处理每个子文件夹
for folder_idx = 1:length(sub_folders)
    current_folder = fullfile(main_folder, sub_folders(folder_idx).name);
    
    % 在主结果文件夹中创建对应的子结果文件夹
    output_folder = fullfile(output_main_folder, sprintf('result%d', folder_idx));
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    % 获取当前子文件夹中的所有.mat文件
    file_list = dir(fullfile(current_folder, '*.mat'));
    
    % 创建结果存储结构
    all_results = struct();
    
    % 创建Excel表格数据
    excel_data = cell(length(file_list)+1, 5);
    excel_data{1,1} = '数据组';
    excel_data{1,2} = 'Y轴均值';
    excel_data{1,3} = 'X/Y轴均值';
    excel_data{1,4} = 'Y/X轴均值';
    excel_data{1,5} = 'Total均值';
    
    % 循环处理当前子文件夹中的每组数据
    for file_idx = 1:length(file_list)
        % 加载当前.mat文件
        filename = fullfile(current_folder, file_list(file_idx).name);
        data = load(filename);
        
        % 提取数据
        time_axis = data.time_axis;
        processed_data = data.processed_data;
        
        % 对第一列数据减去重力加速度
        processed_data(:,1) = processed_data(:,1) - 9.8;
        
        % 1. 计算Total信号（取绝对值）
        total_signal = abs(sqrt(sum(processed_data.^2, 2)));
        total_mean = mean(total_signal);
        
        % 2. 对各通道数据取绝对值
        abs_data = abs(processed_data);
        abs_means = mean(abs_data, 1);
        
        % 存储结果
        all_results(file_idx).abs_means = abs_means;
        all_results(file_idx).total_mean = total_mean;
        
        % 填充Excel数据
        excel_data{file_idx+1,1} = ['第' num2str(file_idx) '组'];
        excel_data{file_idx+1,2} = abs_means(1);
        excel_data{file_idx+1,3} = abs_means(2);
        excel_data{file_idx+1,4} = abs_means(3);
        excel_data{file_idx+1,5} = total_mean;
        
        % 打印当前组的结果
        fprintf('\n===== %s - 第%d组数据结果 =====\n', sub_folders(folder_idx).name, file_idx);
        fprintf('绝对值后的均值:\n Y轴: %.4f\n X/Y轴: %.4f\n Y/X轴: %.4f\n', ...
                abs_means(1), abs_means(2), abs_means(3));
        fprintf('Total均值 (绝对值后): %.4f\n', total_mean);
        
        % 绘制当前组的图像
        if isvector(time_axis) && ismatrix(processed_data)
            % 绘制各通道绝对值数据
            for ch_idx = 1:3
                figure('Position', [100+100*file_idx 100+100*ch_idx 800 400]);
                plot(time_axis, abs_data(:, ch_idx), 'Color', line_colors{ch_idx}, 'LineWidth', 1.5);
                
                % 根据通道设置标题
                switch ch_idx
                    case 1
                        title_str = 'Y轴';
                        save_name = 'Y_axis';
                    case 2
                        title_str = 'X_Y轴';
                        save_name = 'X_Y_axis';
                    case 3
                        title_str = 'Y_X轴';
                        save_name = 'Y_X_axis';
                end
                title(sprintf('%s - 第%d组数据 - %s (绝对值后)', sub_folders(folder_idx).name, file_idx, title_str), 'FontWeight', 'bold');
                xlabel('Time'); 
                ylabel('Amplitude');
                grid on;
                set(gca, 'FontSize', 9, 'Box', 'off');
                
                % 保存图像
                save_path = fullfile(output_folder, sprintf('Group%d_%s.png', file_idx, save_name));
                saveas(gcf, save_path);
                close(gcf);
            end
            
            % 绘制绝对值后的Total信号
            figure('Position', [100+100*file_idx 500 800 400]);
            plot(time_axis, total_signal, 'Color', line_colors{4}, 'LineWidth', 1.5);
            title(sprintf('%s - 第%d组数据 - Total信号 (绝对值后)', sub_folders(folder_idx).name, file_idx), 'FontWeight', 'bold');
            xlabel('Time');
            ylabel('Amplitude');
            grid on;
            set(gca, 'FontSize', 9, 'Box', 'off');
            
            % 保存图像
            save_path = fullfile(output_folder, sprintf('Group%d_Total_abs.png', file_idx));
            saveas(gcf, save_path);
            close(gcf);
        else
            disp(['第' num2str(file_idx) '组数据维度错误，请检查time_axis和processed_data']);
        end
    end
    
    % 将数据写入Excel文件
    excel_path = fullfile(output_folder, '实验结果汇总.xlsx');
    writecell(excel_data, excel_path);
    fprintf('\n结果已保存到文件: %s\n', excel_path);
    
    % 汇总打印当前子文件夹的所有结果
    fprintf('\n===== %s - 最终汇总结果 =====\n', sub_folders(folder_idx).name);
    for file_idx = 1:length(file_list)
        fprintf('\n第%d组数据:\n', file_idx);
        fprintf('  Y轴上的均值 (绝对值后): %.4f\n', all_results(file_idx).abs_means(1));
        fprintf('  X/Y轴均值 (绝对值后): %.4f\n', all_results(file_idx).abs_means(2));
        fprintf('  Y/X轴均值 (绝对值后): %.4f\n', all_results(file_idx).abs_means(3));
        fprintf('  Total均值 (绝对值后): %.4f\n', all_results(file_idx).total_mean);
    end
end

fprintf('\n===== 所有子文件夹处理完成 =====\n');
disp(['所有结果已保存在: ' output_main_folder]);