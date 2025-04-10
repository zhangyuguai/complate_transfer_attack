import argparse
from tqdm import tqdm
import transferattack
from transferattack.utils import *
import os
import requests
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict


def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='mifgsm', type=str, help='the attack algorithm', choices=transferattack.attack_zoo.keys())
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=8, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet18', type=str, help='the source surrogate model')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='0', type=str)
    parser.add_argument('--eval_runs', type=int, default=10,help='需要评估的攻击运行次数')
    return parser.parse_args()

def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['TORCH_HUB_URL'] = "https://mirrors.aliyun.com/pytorch-wheels/torchhub"

    # 创建基础输出目录
    base_output_dir =args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    if not args.eval:
        # 执行10次攻击循环
        for attack_run in range(1, 11):
            # 动态生成当前运行的输出路径
            current_output_dir = os.path.join(
                base_output_dir, 
                f"{args.attack}_{attack_run}"
            )
            os.makedirs(current_output_dir, exist_ok=True)
            # 重新初始化数据加载器（保证每次运行数据顺序一致）
            dataset = AdvDataset(
                input_dir=args.input_dir,
                output_dir=current_output_dir,
                targeted=args.targeted,
                eval=args.eval
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=args.batchsize,
                shuffle=False,
                num_workers=4
            )
            # 初始化攻击器（重要：每次运行重新创建）
            if args.ensemble or len(args.model.split(',')) > 1:
                args.model = args.model.split(',')
            attacker = transferattack.load_attack_class(args.attack)(
                model_name=args.model,
                targeted=args.targeted
            )
            # 攻击生成流程
            for batch_idx, [images, labels, filenames] in tqdm(enumerate(dataloader)):
                if args.attack in ['ttp', 'm3d']:
                    for idx, target_class in tqdm(enumerate(generation_target_classes)):
                        labels = labels - 1
                        perturbations = attacker(images, labels, idx)
                        class_output_dir = os.path.join(
                            current_output_dir, 
                            str(target_class)
                        )
                        os.makedirs(class_output_dir, exist_ok=True)
                        save_images(
                            class_output_dir,
                            images + perturbations.cpu(),
                            filenames
                        )
                else:
                    labels = labels - 1
                    perturbations = attacker(images, labels)
                    save_images(
                        current_output_dir,
                        images + perturbations.cpu(),
                        filenames
                    )
            print(f"第 {attack_run} 次攻击完成，结果保存在：{current_output_dir}")
    else:
        # 增强的评估逻辑
        # 获取所有待评估模型
        model_list = list(load_pretrained_model(cnn_model_paper, vit_model_paper))
        asr_results = defaultdict(list)
        # 遍历所有攻击运行实例
        for run_id in range(1, args.eval_runs + 1):
            current_run_dir = os.path.join(args.output_dir, f"{args.attack}_{run_id}")
            
            if not os.path.exists(current_run_dir):
                print(f"Warning: {current_run_dir} not found, skipping...")
                continue
            # 对每个模型进行评估
            for model_name, model in model_list:
                model = wrap_model(model.eval().cuda())
                model.requires_grad_(False)
                total_asr = 0
                if args.attack in ['ttp', 'm3d']:
                    # 目标攻击评估
                    for target_class in generation_target_classes:
                        target_dir = os.path.join(current_run_dir, str(target_class))
                        dataset = AdvDataset(
                            input_dir=args.input_dir,
                            output_dir=target_dir,
                            targeted=True,
                            target_class=target_class,
                            eval=True
                        )
                        dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)
                        total_asr += eval(model, dataloader, is_targeted=True)
                    avg_asr = total_asr / len(generation_target_classes)
                else:
                    # 非目标攻击评估
                    dataset = AdvDataset(
                        input_dir=args.input_dir,
                        output_dir=current_run_dir,
                        targeted=args.targeted,
                        eval=True
                    )
                    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False)
                    avg_asr = eval(model, dataloader, args.targeted)
                asr_results[model_name].append(avg_asr)
                print(f"[Run {run_id}] {model_name} ASR: {avg_asr:.2f}%")
        
        # 生成统计报告
        print("\n=== Final Evaluation Report ===")
        
        # 1. 首先显示详细统计信息
        for model_name, results in asr_results.items():
            if len(results) == 0:
                continue
                
            mean = np.mean(results)
            std = np.std(results)
            max_asr = np.max(results)
            min_asr = np.min(results)
            
            print(f"\nModel: {model_name}")
            print(f"Average ASR: {mean:.2f}% ± {std:.2f}")
            print(f"Max ASR: {max_asr:.2f}%")
            print(f"Min ASR: {min_asr:.2f}%")
            print(f"All Results: {[round(x,2) for x in results]}")
        
        # 2. 再以列表形式显示每个模型的平均ASR
        mean_asr_list = [round(np.mean(results), 2) for model_name, results in asr_results.items() if len(results) > 0]
        model_names = [model_name for model_name, results in asr_results.items() if len(results) > 0]
        
        print("\n=== 每个模型的平均ASR列表 ===")
        print(f"{mean_asr_list}")
        
        # 3. 显示模型名称与对应的ASR
        print("\n=== 模型名称及对应的平均ASR ===")
        for model_name, mean_asr in zip(model_names, mean_asr_list):
            print(f"{model_name}: {mean_asr}")

def eval(model, dataloader, is_targeted=False):
    correct, total = 0, 0
    model.eval()

    with torch.no_grad():
        for step, (images, labels, filenames) in enumerate(dataloader):
            # Ensure labels are 0-based if they are originally 1-based
            labels = labels - 1
            if is_targeted:
                # If your data has shape [2, batch_size], label[1] is the target
                labels = labels[1]

            # Get predictions
            pred = model(images.cuda()).argmax(dim=1).cpu()

            # Tally correct
            if is_targeted:
                # We want pred == target_label
                correct += (pred == labels).sum().item()
            else:
                # We want pred != original_label
                correct += (pred != labels).sum().item()

            total += labels.size(0)

    # Calculate attack success rate
    asr = correct / total * 100
    print(f"Attack Success Rate: {asr:.2f}%")
    return asr

if __name__ == '__main__':
    main()