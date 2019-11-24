package come.example.qiaowenhao.leetcodepractices;

import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

public class OfferActivity extends AppCompatActivity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_offer);
    }

    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> res = new ArrayList<>();
        if (input == null || input.length < k) {
            return null;
        }

        PriorityQueue<Integer> queue = new PriorityQueue<>(k, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }
        });
        for (int i = 0; i < input.length; i++) {
            if (i < k) {
                queue.offer(input[i]);
            } else if (queue.peek() > input[i]) {
                queue.poll();
                queue.offer(input[i]);
            }
        }

        while (!queue.isEmpty()) {
            res.add(queue.poll());
        }
        return res;
    }

    public int NumberOf1Between1AndN_Solution(int n) {
        int count = 0;
        int index = 1;
        while (n / index != 0) {
            int a = n / index;
            int b = n % index;
            count += (n + 8) * index + b;
        }
        return count;
    }

    public int GetNumberOfK(int [] array , int k) {
        if (array == null || array.length == 0) {
            return 0;
        }
        return binarySearch(array, k + 0.5) - binarySearch(array, k - 0.5);

    }

    private int binarySearch(int[] array, double target) {
        int start = 0;
        int end = array.length - 1;
        while (start <= end) {
            int mid = (start + end) / 2;
            if (target > array[mid]) {
                start = mid + 1;
            } else if (target < array[mid]) {
                end = mid - 1;
            }
        }
        return start;
    }

    public String PrintMinNumber(int [] numbers) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < numbers.length; i++) {
            list.add(numbers[i]);
        }

        Collections.sort(list, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                String s1 = o1 + "" + o2;
                String s2 = o2 + "" + o1;
                return s1.compareTo(s2);
            }
        });

        String str = "";
        for (int i = 0; i < list.size(); i++) {
            str += list.get(i);
        }
        return str;
    }

    public int numDecodings(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int n = s.length();
        int[] dp = new int[n];
        dp[0] = 0;
        dp[1] = s.charAt(0) >= '1' && s.charAt(0) <= '9' ? 1 : 0;
        for (int i = 2; i <= n; i++) {
            String str1 = s.substring(i - 1, i);
            int value1 = Integer.valueOf(str1);
            if (value1 >= 1 && value1 <= 9) {
                dp[i] += dp[i - 1];
            }
            String str2 = s.substring(i - 2, i);
            int value2 = Integer.valueOf(str2);
            if (value2 >= 10 && value2 <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[n];
    }

    public int GetUglyNumber_Solution(int index) {
        if (index == 1) {
            return 1;
        }
        int[] nums = new int[index + 1];
        nums[1] = 1;
        int id2 = 1, id3 = 1, id5 = 1;
        for (int i = 2; i <= index; i++) {
            nums[i] = Math.min(nums[id2] * 2, Math.min(nums[id3] * 3, nums[id5] * 5));
            if (nums[i] == nums[id2] * 2) {
                id2++;
            }
            if (nums[i] == nums[id3] * 3) {
                id3++;
            }
            if (nums[i] == nums[id5] * 5) {
                id5++;
            }
        }
        return nums[index];
    }

    public int FirstNotRepeatingChar(String str) {
        if (str == null) {
            return -1;
        }

        char[] chars = str.toCharArray();
        int length = chars.length;
        int[] nums = new int[256];
        for (int i = 0; i < length; i++) {
            nums[hash(chars[i])] ++;
        }

        int index = 0;
        for (int i = 0; i < length; i++) {
            if(nums[hash(chars[i])] == 1) {
                index = i;
                break;
            }
        }

        return index;
    }

    private int hash(char c) {
        return c + 128;
    }


    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        int len1 = getLength(pHead1);
        int len2 = getLength(pHead2);
        if (len1 > len2) {
            pHead1 = fastWalk(len1 - len2, pHead1);
        }
        if (len2 > len1) {
            pHead2 = fastWalk(len2 - len1, pHead2);
        }

        while (pHead1 != null && pHead2 != null) {
            if (pHead1.val == pHead2.val) {
                return pHead1;
            }
            pHead1 = pHead1.next;
            pHead2 = pHead2.next;
        }
        return null;
    }

    private int getLength(ListNode node) {
        int length = 0;
        while (node != null) {
            length++;
            node = node.next;
        }
        return length;
    }

    private ListNode fastWalk(int gap, ListNode node) {
        while (gap-- > 0) {
            node = node.next;
        }
        return node;
    }

    public void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {
        int result = 0;
        for (int i = 0; i < array.length; i++) {
            result ^= array[i];
        }
        int index = findIndex(result);

        for (int i = 0; i < array.length; i++) {
            if (shouldInNum1(array[i], index)) {
                num1[0] ^= array[i];
            } else {
                num2[0] ^= array[i];
            }
        }
    }

    private int findIndex(int num) {
        int index = 0;
        while((num & 1) == 0) {
            index++;
            num = num >> 1;
        }
        return index;
    }

    private boolean shouldInNum1(int num, int index) {
        return ((num >> index) & 1) == 1;
    }

    public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
        ArrayList<Integer> res = new ArrayList<Integer>();
        if (array == null || array.length == 0) {
            return res;
        }
        int start = 0;
        int end = array.length;
        while (start < end) {
            if (array[start] + array[end] < sum) {
                start++;
            } else if (array[start] + array[end] > sum) {
                end--;
            } else {
                res.add(array[start]);
                res.add(array[end]);
                break;
            }
        }
        return res;
    }

    public ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        int start = 1, end = 2;
        while (start < end) {
            int total = (start + end) * (end - start + 1) / 2;
            if (total == sum) {
                ArrayList<Integer> temp = new ArrayList<>();
                for (int i = start; i <= end; i++) {
                    temp.add(i);
                }
                res.add(temp);
                start++;
            } else if (total < sum) {
                end++;
            } else {
                start++;
            }
        }
        return res;
    }


    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;

        }

    }

    int index = 0;
    TreeNode KthNode(TreeNode pRoot, int k)
    {
        if (pRoot == null) {
            return null;
        }

        TreeNode left = KthNode(pRoot.left, k);
        if (left != null) {
            return left;
        }

        index++;
        if (index == k) {
            return pRoot;
        }

        TreeNode right = KthNode(pRoot.right, k);
        if (right != null) {
            return right;
        }
        return null;
    }

    public int TreeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = TreeDepth(root.left);
        int right = TreeDepth(root.right);
        return Math.max(left, right) + 1;
    }

    private int getTreeDepth(TreeNode root, int k) {
        if (root == null) {
            return k;
        }
        int left = 0, right = 0;
        if (root.left != null) {
            left = getTreeDepth(root.left, k++);
        }
        if (root.right != null) {
            right = getTreeDepth(root.right, k++);
        }
        return Math.max(left, right);
    }

    public boolean IsBalanced_Solution(TreeNode root) {
        return getDepth(root) != -1;

    }

    private int getDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int left = getDepth(root.left);
        if (left == -1) {
            return -1;
        }
        int right = getDepth(root.right);
        if (right == -1) {
            return -1;
        }
        return Math.abs(left - right) > 1 ? -1 : Math.max(left, right) + 1;
    }

    public String ReverseSentence(String str) {
        char[] chars = str.toCharArray();
        int length = chars.length;
        reverse(chars, 0, length - 1);
        int i = 0, start = 0, end = 0;
        while (i < length) {
            while (i < length && chars[i] == ' ') {
                i++;
            }
            start = end = i;
            while (i < length && chars[i] != ' ') {
                end++;
                i++;
            }
            reverse(chars, start, end - 1);
        }
        return String.valueOf(chars);
    }

    private void reverse(char[] chars, int start, int end) {
        while (start < end) {
            char tmp = chars[start];
            chars[start] = chars[end];
            chars[end] = tmp;
            start++;
            end--;
        }
    }

    public String LeftRotateString(String str,int n) {
        if (n == 0 || str == null || str.length() == 0) {
            return str;
        }

        char[] chars = str.toCharArray();
        int length = chars.length;
        n = n % length;
        reverse(chars, 0, n - 1);
        reverse(chars, n, length - 1);
        reverse(chars, 0, length - 1);
        return String.valueOf(chars);
    }

    public ArrayList<Integer> maxInWindows(int [] num, int size) {
        ArrayList<Integer> res = new ArrayList<>();
        if (num == null || size <= 0) {
            return res;
        }
        Deque<Integer> deque = new ArrayDeque<Integer>();
        for (int i = 0; i < num.length; i++) {
            while (!deque.isEmpty() && deque.peek() < i - size + 1) {
                deque.poll();
            }
            while (!deque.isEmpty() && num[deque.peek()] < num[i]) {
                deque.pollLast();
            }
            deque.offer(i);
            if (i + 1 >= size) {
                res.add(num[deque.peek()]);
            }
        }
        return res;
    }

    public boolean isContinuous(int [] numbers) {
        if (numbers == null || numbers.length < 5) {
            return false;
        }
        int max = 0, min = 0, flag = 0;
        for (int i = 0; i < numbers.length; i++) {
            int tmp = numbers[i];
            if(tmp == 0) {
                continue;
            }
            if (tmp < 0 || tmp > 13) {
                return false;
            }
            if (((flag >> tmp) & 1) == 1) {
                return false;
            }

            max = Math.max(tmp, max);
            min = Math.min(tmp, min);
            flag |= (1 << tmp);
        }
        return max - min < 5;
    }

    public boolean Find(int target, int [][] array) {
        int line = array.length;
        int row = array[0].length;
        for (int i = line - 1, j = 0; i >= 0 && j < row;) {
            int tmp = array[i][j];
            if (target == tmp) {
                return true;
            } else if (target < tmp) {
                i--;
                continue;
            } else {
                j++;
                continue;
            }
        }
        return false;
    }

    public String replaceSpace(StringBuffer str) {
        int count = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == ' ') {
                count++;
            }
        }

        int oldLength = str.length();
        int newLength = oldLength + count * 2;
        int newIndex = newLength - 1;
        str.setLength(newLength);
        for (int i = oldLength - 1; i>= 0 && i < newIndex; i--) {
            if (str.charAt(i) == ' ') {
                str.setCharAt(newIndex--, '0');
                str.setCharAt(newIndex--, '2');
                str.setCharAt(newIndex--, '%');
            } else {
                str.setCharAt(newIndex--, str.charAt(i));
            }
        }
        return str.toString();
    }

    ArrayList<Integer> res = new ArrayList<>();

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if (listNode != null) {
            printListFromTailToHead(listNode.next);
            res.add(listNode.val);
        }
        return res;
    }

    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        return reConstruct(pre, 0, pre.length - 1, in,0, in.length - 1);
    }


    public TreeNode reConstruct(int[] pre, int preStart, int preEnd,int[] in, int inStart, int inEnd) {
        if (preStart > preEnd || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(pre[preStart]);
        for (int i = inStart; i<= inEnd; i++) {
            if (in[i] == pre[preStart]) {
                root.left = reConstruct(pre, preStart + 1, i - inStart + preStart, in, inStart, i - 1);
                root.right = reConstruct(pre, i - inStart + preStart + 1, preEnd, in, i + 1, inEnd);
                break;
            }
        }
        return root;
    }

    public int minNumberInRotateArray(int [] array) {
        int low = 0, high = array.length - 1;
        while (low < high) {
            int mid = (low + high) >> 1;
            if (array[mid] > array[high]) {
                low = mid + 1;
            } else if (array[mid] == array[high]) {
                high = high - 1;
            } else {
                high = mid;
            }
        }
        return array[low];
    }

    public int Fibonacci(int n) {
        if (n <= 1) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    public int JumpFloor(int target) {
        if (target <= 2) {
            return target;
        }
        int[] dp = new int[target+1];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= target; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[target];
    }

    public int JumpFloorII(int target) {
        if (target <= 1) {
            return target;
        }
        int[] res = new int[target + 1];
        res[0] = 1;
        res[1] = 1;
        for (int i = 2; i <= target; i++) {
            res[i] = 2 * res[i - 1];
        }
        return res[target];
    }

    public int RectCover(int target) {
        if (target <= 2) {
            return target;
        }
        int[] res = new int[target + 1];
        res[1] = 1;
        res[2] = 2;
        for (int i = 3; i <= target; i++) {
            res[i] = res[i - 1] + res[i - 2];
        }
        return res[target];
    }

    public int NumberOf1(int n) {
        int count = 0;
        while (n != 0) {
            count++;
            n = n & (n - 1);
        }
        return count;
    }

    public double Power(double base, int exponent) {
        int n = exponent;
        double res = 1;
        if (exponent == 0) {
            return 1;
        } else if (exponent < 0) {
            if (base == 0) {
                throw new RuntimeException("分母不能为0");
            }
            n = -n;
        }
        while (n !=0) {
            if ((n & 1) == 1) {
                res = res * base;
            }
            n = n >> 1;
            base *= base;
        }

        return exponent > 0 ? res : 1 / res;
    }

    public ArrayList<String> Permutation(String str) {
        ArrayList<String> res = new ArrayList<>();
        if (str == null) {
            return res;
        }
        permutation(str.toCharArray(), 0, res);
        Collections.sort(res);
        return res;
    }

    public void permutation(char[] chars, int start, ArrayList<String> res) {
        if (start == chars.length - 1) {
            String temp = String.valueOf(chars);
            if (!res.contains(temp)) {
                res.add(temp);
            }
        } else {
            for (int i = start; i < chars.length; i++) {
                swap(chars, start, i);
                permutation(chars, start+1, res);
                //在递归结束之后加入list之后再交换回来
                swap(chars, start, i);
            }
        }
    }

    private void swap(char[] chars, int start, int end) {
        char tmp = chars[start];
        chars[start] = chars[end];
        chars[end] = tmp;
    }

    public int Sum_Solution(int n) {
        int sum = n;
        boolean hasResult = (n > 0) && ((sum += Sum_Solution(n - 1)) > 0);
        return sum;
    }

    public int Add(int num1,int num2) {
        int sum = num1 ^ num2;
        int add = (num1 & num2) << 1;
        while (add != 0) {
            sum = sum ^ add;
            add = (sum & add) << 1;
        }
        return sum;
    }

    public int StrToInt(String str) {
        int length = str.length();
        char[] chars = str.toCharArray();
        int i = 0;
        int res = 0;
        int sign = 1;
        while (i < length && chars[i] ==' ') {
            i++;
        }
        if (i < length && (chars[i] == '-' || chars[i] == '+')) {
            sign = chars[i] == '-' ? -1 : 1;
            i++;
        }
        for (; i < length; i++) {
            int value = chars[i] - '0';
            if (value < 0 || value > 9) {
                res = 0;
                break;
            }
            value = value * sign;
            if (sign == 1 && (res > Integer.MAX_VALUE / 10 || (res == Integer.MAX_VALUE / 10 && value > Integer.MAX_VALUE % 10))) {
                return 0;
            }

            if (sign == -1 && (res < Integer.MIN_VALUE / 10 || (res == Integer.MIN_VALUE / 10 && Math.abs(value) > Integer.MAX_VALUE % 10 + 1))) {
                return 0;
            }
            res = res * 10;
            res += value;
        }
        return res;
    }

    public boolean duplicate(int numbers[],int length,int [] duplication) {
        if (numbers == null || length == 0) {
            return false;
        }
        boolean[] dups = new boolean[length];
        for (int i = 0; i < length; i++) {
            if (dups[numbers[i]] == false) {
                dups[numbers[i]] = true;
            } else {
                duplication[0] = numbers[i];
                return true;
            }
        }
        return false;
    }

    public int[] multiply(int[] A) {
        int length = A.length;
        int[] left = new int[length];
        int[] res = new int[length];
        left[0] = 1;
        for (int i = 1; i < length; i++) {
            left[i] = left[i - 1] * A[i - 1];
        }
        int[] right = new int[length];
        right[length - 1] = 1;
        for (int i = length - 2; i >=0; i--) {
            right[i] = right[i+1] * A[i+1];
        }
        for (int i = 0; i < length; i++) {
            res[i] = left[i] * right[i];
        }
        return res;
    }

    public boolean isNumeric(char[] str) {
        int length = str.length;
        boolean hasDot = false;
        boolean hasE = false;
        for (int i = 0; i < length; i++) {
            char cur = str[i];
            if (isSign(cur)) {
                if (i !=0 && !((i - 1) > 0 && isE(str[i-1]))) {
                    return false;
                }
            } else if (isDot(cur)) {
                if (hasDot) {
                    return false;
                }
                if (hasE) {
                    return false;
                }
                hasDot = true;
            } else if (isE(cur)) {
                hasE = true;
                if (i == length-1) {
                    return false;
                }
            } else if (!isNumber(cur)) {
                return false;
            }
        }

        return true;
    }

    private boolean isE(char ch) {
        return ch == 'e' || ch == 'E';
    }

    private boolean isSign(char ch) {
        return ch == '+' || ch == '-';
    }

    private boolean isDot(char ch) {
        return ch == '.';
    }

    private boolean isNumber(char ch) {
        return ch - '0' >= 0 && ch - '0' <= 9;
    }

    int[] nums = new int[256];
    Queue<Character> queue = new LinkedList<>();

    public void Insert(char ch)
    {
        nums[ch]++;
        if (nums[ch] == 1) {
            queue.offer(ch);
        }
    }
    //return the first appearence once char in current stringstream
    public char FirstAppearingOnce()
    {
        while (!queue.isEmpty() && nums[queue.peek()] > 1) {
            queue.poll();
        }
        return queue.isEmpty() ? '#' : queue.peek();
    }


    public ListNode deleteDuplication(ListNode pHead)
    {
        ListNode helper = new ListNode(-1);
        helper.next = pHead;
        ListNode fast = pHead;
        ListNode slow = helper;
        while (fast != null) {
            if (fast.next != null && fast.val != fast.next.val) {
                fast = fast.next;
                slow = slow.next;
            } else {
                while (fast.next != null && fast.val == fast.next.val) {
                    fast = fast.next;
                }
                slow.next = fast.next;
                fast = fast.next;
            }
        }

        return helper.next;

    }

    public class TreeLinkNode {
        int val;
        TreeLinkNode left = null;
        TreeLinkNode right = null;
        TreeLinkNode next = null;

        TreeLinkNode(int val) {
            this.val = val;
        }
    }

    public TreeLinkNode GetNext(TreeLinkNode pNode)
    {
        TreeLinkNode node = null;
        if (pNode.right != null) {
            node = pNode.right;
            while (node.left != null) {
                node = node.left;
            }
            return node;
        } else {
            node = pNode.next;
            while (node.next != null) {
                if (node.next.left == node) {
                    return node.next;
                }
                node = node.next;
            }
            return null;
        }
    }

    public boolean hasPath(char[] matrix, int rows, int cols, char[] str)
    {
        int[] flag = new int[matrix.length];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (hasPath(matrix, flag, i, rows, j, cols, str, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean hasPath(char[] matrix, int[] flag, int i, int rows, int j, int cols, char[] str, int cur) {
        int index = i * cols + j;
        if (i < 0 || i >= rows || j < 0 || j >= cols || matrix[index] != str[cur] || flag[index] == 1) {
            return false;
        }

        if (cur == str.length - 1) {
            return true;
        }

        flag[index] = 1;

        //　注意不能使用自加++
        if (hasPath(matrix, flag, i - 1, rows, j, cols, str, cur + 1)
                || hasPath(matrix, flag, i, rows, j - 1, cols, str, cur + 1)
                || hasPath(matrix, flag, i + 1, rows, j, cols, str, cur + 1)
                || hasPath(matrix, flag, i, rows, j + 1, cols, str, cur + 1)) {
            return true;
        }
        flag[index] = 0;
        return false;
    }


    public int movingCount(int threshold, int rows, int cols)
    {
        boolean[][] flag = new boolean[rows][cols];
        return movingCount(threshold, 0, rows, 0, cols, flag);
    }

    public int movingCount(int threshold, int i, int rows, int j, int cols, boolean[][] flag) {
        if (i < 0 || i >= rows || j < 0 || j >= cols || flag[i][j] == true || countNums(i) + countNums(j) > threshold) {
            return 0;
        }
        flag[i][j] = true;
        return movingCount(threshold, i - 1, rows, j, cols, flag)
                + movingCount(threshold, i + 1, rows, j, cols, flag)
                + movingCount(threshold, i, rows, j - 1, cols, flag)
                + movingCount(threshold, i, rows, j + 1, cols, flag) + 1;
    }

    private int countNums(int num) {
        int res = 0;
        while (num % 10 != 0) {
            res += num % 10;
            num = num / 10;
        }
        return res;
    }

    public int cutRope(int target) {
        if (target == 2) {
            return 1;
        }
        if (target == 3) {
            return 2;
        }
        int[] dp = new int[target + 1];
        /*
        下面3行是n>=4的情况，跟n<=3不同，4可以分很多段，比如分成1、3，
        这里的3可以不需要再分了，因为3分段最大才2，不分就是3。记录最大的。
         */
        dp[1] = 1;
        dp[2] = 2;
        dp[3] = 3;
        int res = 0;
        for (int i = 4; i <= target; i++) {
            for (int j = 1; j <= i/ 2; j++) {
                res = Math.max(res, dp[j] * dp[i - j]);
            }
        }
        return res;
    }



    public boolean match(char[] str, char[] pattern)
    {
        return helper(str, pattern, 0, 0);
        //return matchCore(str, 0, pattern, 0);
    }

    public boolean helper(char[] str, char[] pattern, int i, int j) {
        // 匹配成功的条件，递归终止
        if (i == str.length && j == pattern.length) {
            return true;
        }
        /*if (i > str.length && j >= pattern.length) {
            return false;
        }*/

        // 模式第二个字符是*
        if ((j + 1) < pattern.length && pattern[j + 1] == '*') {
            // 模式第二个字符跟字符串第一个字符相匹配，分三种模式
            if (i < str.length && (str[i] == pattern[j] || pattern[j] == '.')) {
                return helper(str, pattern, i, j + 2)//　模式后移2,视为x*匹配0个字符
                        || helper(str, pattern, i + 1, j + 2)//　模式匹配一个字符
                        || helper(str, pattern, i + 1, j);// 匹配一个，再匹配字符串中的下一个
            } else {
                return helper(str, pattern, i, j + 2);
            }
        }
        // 模式第二个字符不是*,能匹配上则后移，否则返回false
        if ((i < str.length && j < pattern.length && str[i] == pattern[j]) || (j < pattern.length && pattern[j] == '.' && i < str.length)) {
            return helper(str, pattern, i + 1, j + 1);
        }
        return false;

    }
}
