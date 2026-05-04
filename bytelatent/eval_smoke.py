import torch

from bytelatent.distributed import DistributedArgs, dist_sum, get_local_rank, setup_torch_distributed


def main():
    dist_args = DistributedArgs()
    dist_args.configure_world()
    setup_torch_distributed(dist_args)

    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    reduced_bytes = dist_sum(1, reduce_dtype=torch.bfloat16)
    reduced_loss = dist_sum(1.5, reduce_dtype=torch.bfloat16)

    if torch.distributed.get_rank() == 0:
        print(
            {
                "world_size": torch.distributed.get_world_size(),
                "reduced_bytes": reduced_bytes.item(),
                "reduced_loss": reduced_loss.item(),
            }
        )

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
