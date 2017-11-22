function [pm, pf, pb, det_ap, cls_ap] = compute_evaluations(img_list, allboxes, time)
    % 加载标定数据
    addpath('.\core\imdb\cache');
    load imdb_voc_2007_test;
    load roidb_voc_2007_test_easy;
    % 计算目标总数、虚警漏警个数和正确识别的个数
    target_cnt = 0;
    loujing_cnt = 0;
    xujing_cnt = 0;
    reg_cnt = 0;
    for i = 1:length(img_list)
        index = find(strcmp(imdb.image_ids, char(img_list(i))));
        % 标定位置
        gt_boxes = roidb.rois(index).boxes;
        % 标定类别
        gt_class = roidb.rois(index).class;
        % 标定总数
        gt_num = length(gt_class);
        target_cnt = target_cnt + gt_num;
        % 预测的位置、分数和类别
        if ~size(allboxes{i}, 1)
            continue
        end
        pred_boxes = allboxes{i}(:, 1:4);
        pred_score = allboxes{i}(:, 5);
        pred_class = allboxes{i}(:, 6);
        % 预测总数
        pred_num = size(allboxes{i}, 1);
        
        loujing_flag = ones(1, gt_num);
        xujing_flag = ones(1, pred_num);
        reg_flag = ones(1, gt_num);

        max_overlap = ones(1, pred_num) * -inf;
        reg_boxes = zeros(1, gt_num);
        for j = 1:pred_num
            for k = 1:gt_num
                overlap = compute_overlap(pred_boxes(j, :), gt_boxes(k, :));
                % 覆盖大于0.5，说明检测位置正确
                if overlap >= 0.5
                    % 有真实目标可以匹配的，这个区域就不是虚警
                    xujing_flag(j) = false;
                    % 有真实目标可以匹配的，这个真实目标不会被漏警
                    loujing_flag(k) = false;
                    % 如果最后还是有重复框，取overlap最大的做判决
                    if overlap > max_overlap(j)
                        max_overlap(j) = overlap;
                        reg_boxes(k) = pred_class(j);
                    end
                end
            end
        end
        loujing_cnt = loujing_cnt + sum(loujing_flag);
        xujing_cnt = xujing_cnt + sum(xujing_flag);
        reg_cnt = reg_cnt + sum(reg_boxes == gt_class');
    end
    pm = loujing_cnt / target_cnt * 100;
    pf = xujing_cnt / target_cnt * 100;
    pb = reg_cnt / (target_cnt - loujing_cnt) * 100;
    
    fprintf('--------------------------\n');
    fprintf('Mean time used: %.3fs\n', time / length(img_list));
    fprintf('Real target number: %d\n', target_cnt);
    fprintf('Missing alarm: %.1f%%\n', pm);
    fprintf('False alarm: %.1f%%\n', pf);
    fprintf('Recognition rate: %.1f%%\n', pb);
end

function overlap = compute_overlap(box1, box2)
    x1 = box1(1);
    y1 = box1(2);
    x2 = box1(3);
    y2 = box1(4);
    area1 = (x2-x1+1) * (y2-y1+1);
    x1 = box2(1);
    y1 = box2(2);
    x2 = box2(3);
    y2 = box2(4);
    area2 = (x2-x1+1) * (y2-y1+1);
    xx1 = max(x1, box1(1));
    yy1 = max(y1, box1(2));
    xx2 = min(x2, box1(3));
    yy2 = min(y2, box1(4));

    w = max(0.0, xx2-xx1+1);
    h = max(0.0, yy2-yy1+1);

    inter = w*h;
    overlap = inter ./ ((area1 + area2) - inter);
end