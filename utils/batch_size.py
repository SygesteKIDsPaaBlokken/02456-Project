from torch.cuda import get_device_properties

def get_batch_size_from_vram():
    vram = get_device_properties('cuda:0').total_memory
    
    if vram < 16e9:
        raise MemoryError('Too litle VRAM')
    
    if vram < 32e9:
        return 64
    
    if vram < 64e9:
        return 128
    
    return 256