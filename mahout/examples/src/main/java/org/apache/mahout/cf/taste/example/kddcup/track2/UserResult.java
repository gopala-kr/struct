/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.example.kddcup.track2;

final class UserResult {

  private final long userID;
  private final byte[] resultBytes;

  UserResult(long userID, boolean[] result) {

    this.userID = userID;

    int trueCount = 0;
    for (boolean b : result) {
      if (b) {
        trueCount++;
      }
    }
    if (trueCount != 3) {
      throw new IllegalStateException();
    }

    resultBytes = new byte[result.length];
    for (int i = 0; i < result.length; i++) {
      resultBytes[i] = (byte) (result[i] ? '1' : '0');
    }
  }

  public long getUserID() {
    return userID;
  }

  public byte[] getResultBytes() {
    return resultBytes;
  }


}
