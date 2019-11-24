package come.example.qiaowenhao.leetcodepractices;

import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

public class LeetCodeActivity extends AppCompatActivity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_leetcode);
    }

    // 349. Intersection of Two Arrays
    // ****
    public int[] intersection(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i = 0, j = 0;
        Set<Integer> set = new HashSet<Integer>();
        while (i < nums1.length && j < nums2.length) {
            if (nums1[i] < nums2[j]) {
                i++;
            } else if (nums1[i] > nums2[j]) {
                j++;
            } else {
                set.add(nums1[i]);
                i++;
                j++;
            }

        }
        int[] res = new int[set.size()];
        int index = 0;
        Iterator iterator = set.iterator();
        while (iterator.hasNext()) {
            res[index++] = ((Integer) iterator.next());
        }
        return res;
    }

    // 75. Sort Colors
    // ****
    public void sortColors(int[] nums) {
        int start = 0, end = nums.length - 1;
        for (int i = 0; i <= end; i++) {
            while (nums[i] == 2 && i < end) {
                swap(nums, i, end--);
            }
            while (nums[i] == 0 && i > start) {
                swap(nums, i, start++);
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[j];
        nums[j] = nums[i];
        nums[i] = temp;
    }


    public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    // 61. Rotate List
    //****
    public ListNode rotateRight(ListNode head, int k) {
        int length = getLength(head);
        //注意边界条件
        if (length == 0|| k == 0 || k % getLength(head) == 0) {
            return null;
        }
        k = k % getLength(head);
        ListNode newHead = new ListNode(-1);
        newHead.next = head;
        ListNode slow = newHead;
        ListNode fast = newHead;
        int n = k;
        while (n-- > 0) {
            fast = fast.next;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        newHead.next = slow.next;
        slow.next = null;
        fast.next = head;
        return newHead.next;
    }

    private int getLength(ListNode node) {
        int length = 0;
        while (node != null) {
            node = node.next;
            length++;
        }
        return length;
    }

    // 3. Longest Substring Without Repeating Characters
    //****
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Queue<Character> queue = new LinkedList<>();
        char[] chars = s.toCharArray();
        int length = Integer.MIN_VALUE;
        for (int i = 0; i < chars.length; i++) {
            while (queue.contains(chars[i])) {
                queue.poll();
            }
            queue.add(chars[i]);
            length = Math.max(length, queue.size());
        }
        return length;
    }

    // 11. Container With Most Water
    // **
    public int maxArea(int[] height) {
        int res = 0;
        int start = 0, end = height.length - 1;
        while (start < end) {
            res = Math.max(res, Math.min(height[start], height[end]) * (end - start));
            if (height[start] > height[end]) {
                end--;
            } else {
                start++;
            }
        }
        return res;
    }

    // 3Sum
    // ****
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length < 3) {
            return res;
        }

        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (i == 0 || (i > 0 && nums[i-1] != nums[i])) {
                int start = i + 1, end = nums.length - 1;
                while (start < end) {
                    if (nums[start] + nums[end] == -nums[i]) {
                        res.add(Arrays.asList(nums[i], nums[start], nums[end]));
                        while (start < end && nums[start + 1] == nums[start]) start++;
                        while (start < end && nums[end -1] == nums[end]) end--;
                        start++;
                        end--;
                    } else if (nums[start] + nums[end] < -nums[i]) {
                        start++;
                    } else {
                        end--;
                    }
                }
            }
        }
        return res;
    }

    // 19. Remove Nth Node From End of List
    // ***
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int length = getSize(head);
        if (length == 0 || length < n) {
            return head;
        }
        ListNode helper = new ListNode(-1);
        helper.next = head;
        ListNode fast = helper;
        ListNode slow = helper;
        while (n-- > 0) {
            fast = fast.next;
        }
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = fast;
        return helper.next;
    }

    private int getSize(ListNode listNode) {
        int length = 0;
        while (listNode != null) {
            listNode = listNode.next;
            length++;
        }
        return length;
    }

}
