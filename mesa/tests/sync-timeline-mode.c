/* Compile the pinned Mesa implementation and expose its static mode selector
 * to the offline harness. No substitute model of the selection algorithm. */
#include "vk_device.c"

enum vk_device_timeline_mode
test_native_timeline_mode(struct vk_physical_device *physical)
{
   return get_timeline_mode(physical);
}
