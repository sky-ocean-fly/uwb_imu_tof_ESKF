#include "nlink_protocol.h"

#include <assert.h>

#include <numeric>
#include <string>

//测试添加
#include <ros/ros.h>
//

void NLinkProtocol::HandleData(const uint8_t *data) {
  UnpackFrameData(data);

//   //测试添加
//  ROS_INFO("Hello! I am here.");
//   //

  assert(HandleDataCallback_);
  HandleDataCallback_();
}

bool NLinkProtocol::Verify(const uint8_t *data) {
  uint8_t sum = 0;
  return data[length() - 1] ==
         std::accumulate(data, data + length() - sizeof(sum), sum);
}

bool NLinkProtocolVLength::UpdateLength(const uint8_t *data,
                                        size_t available_bytes) {
  if (available_bytes < 4)
    return false;
  return set_length(static_cast<size_t>(data[2] | data[3] << 8));
}
